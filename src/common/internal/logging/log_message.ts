import { extractImportantStackTrace } from '../stack.js';
import { Logger } from './logger.js';

/**
 * Error with arbitrary `extra` data attached, for debugging.
 * The extra data is omitted if not running the test in debug mode (`?debug=1`).
 */
export class ErrorWithExtra extends Error {
  readonly extra: { [k: string]: unknown };

  /**
   * `extra` function is only called if in debug mode.
   * If an `ErrorWithExtra` is passed, its message is used and its extras are passed through.
   */
  constructor(message: string, extra: () => {});
  constructor(base: ErrorWithExtra, newExtra: () => {});
  constructor(baseOrMessage: string | ErrorWithExtra, newExtra: () => {}) {
    const message = typeof baseOrMessage === 'string' ? baseOrMessage : baseOrMessage.message;
    super(message);

    const oldExtras = baseOrMessage instanceof ErrorWithExtra ? baseOrMessage.extra : {};
    this.extra = Logger.globalDebugMode
      ? { ...oldExtras, ...newExtra() }
      : { omitted: 'pass ?debug=1' };
  }
}

export class LogMessageWithStack extends Error {
  readonly extra: unknown;

  private stackHiddenMessage: string | undefined = undefined;

  constructor(name: string, ex: Error | ErrorWithExtra) {
    super(ex.message);

    this.name = name;
    this.stack = ex.stack;
    if ('extra' in ex) {
      this.extra = ex.extra;
    }
  }

  /** Set a flag so the stack is not printed in toJSON(). */
  setStackHidden(stackHiddenMessage: string) {
    this.stackHiddenMessage ??= stackHiddenMessage;
  }

  toJSON(): string {
    let m = this.name;
    if (this.message) m += ': ' + this.message;
    if (this.stack) {
      if (this.stackHiddenMessage === undefined) {
        m += '\n' + extractImportantStackTrace(this);
      } else if (this.stackHiddenMessage) {
        m += `\n  at (elided: ${this.stackHiddenMessage})`;
      }
    }
    return m;
  }
}

/**
 * Returns a string, nicely indented, for debug logs.
 * This is used in the cmdline and wpt runtimes. In WPT, it shows up in the `*-actual.txt` file.
 */
export function prettyPrintLog(log: LogMessageWithStack): string {
  return '  - ' + log.toJSON().replace(/\n/g, '\n    ');
}
