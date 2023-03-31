export const description = `
Basic tests for sorting.
`;

import { Fixture } from "../common/framework/fixture.js";
import { makeTestGroup } from "../common/framework/test_group.js";

export const g = makeTestGroup(Fixture);

g.test("test")
  .desc("test descr")
  .fn((t) => {
    t.expect(true);
  });
