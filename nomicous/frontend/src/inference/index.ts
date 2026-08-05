/**
 * What the interface knows about **inference host**s.
 *
 * There is no client here, and there is deliberately nothing to add one to.
 * Since ADR 0002 the browser never reaches an **inference agent**: it reads an
 * account setting, it reads **capacity**, and it reads what each job says about
 * where it ran. Everything else happens between the agent and the platform.
 */
export {
  AGENT_INSTALL_COMMAND,
  AGENT_INSTALL_COMMAND_PIP,
  AGENT_PACKAGE_NAME,
  AGENT_PAIR_COMMAND,
  AGENT_RUN_COMMAND,
} from "./constants";
export {
  HOST_PREFERENCE_HINT,
  HOST_PREFERENCE_LABEL,
  useHostPreference,
} from "./hostPreference";
export type { HostPreference } from "./hostPreference";
export {
  executionAnnouncement,
  INFERENCE_HOST_LABEL,
  INFERENCE_HOST_NOUN,
  INFERENCE_HOST_PHRASE,
  isSubmissionRefusal,
  jobExecution,
  submissionRefusalExplanation,
} from "./executionTarget";
export type { JobExecution } from "./executionTarget";
