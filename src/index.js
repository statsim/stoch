import * as distributions from './distributions'
import * as bijectors from './bijectors'
import * as mcmc from './mcmc'
import * as vi from './vi'
import * as math from './math'
import * as stats from './stats'
import * as gp from './gp'

let _globalValidateArgs = true

const setValidateArgs = (value) => {
  _globalValidateArgs = !!value
}

const getValidateArgs = () => _globalValidateArgs

export {
  distributions,
  bijectors,
  mcmc,
  vi,
  math,
  stats,
  gp,
  setValidateArgs,
  getValidateArgs
}

export default {
  distributions,
  bijectors,
  mcmc,
  vi,
  math,
  stats,
  gp,
  setValidateArgs,
  getValidateArgs
}
