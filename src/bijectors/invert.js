import { Bijector } from './bijector'

/**
 * Invert bijector: wraps another bijector, swapping forward and inverse.
 *
 * Forward:  y = inner.inverse(x)
 * Inverse:  x = inner.forward(y)
 * FLDJ:     inner.inverseLogDetJacobian(x)
 * ILDJ:     inner.forwardLogDetJacobian(y)
 */
export class Invert extends Bijector {
  constructor({ bijector, validateArgs, name } = {}) {
    super({
      forwardMinEventNdims: bijector.inverseMinEventNdims,
      inverseMinEventNdims: bijector.forwardMinEventNdims,
      isConstantJacobian: bijector.isConstantJacobian,
      validateArgs,
      name: name || `Invert(${bijector.name})`
    })
    this._bijector = bijector
  }

  get bijector() { return this._bijector }

  _forward(x) { return this._bijector._inverse(x) }
  _inverse(y) { return this._bijector._forward(y) }
  _forwardLogDetJacobian(x) { return this._bijector._inverseLogDetJacobian(x) }
  _inverseLogDetJacobian(y) { return this._bijector._forwardLogDetJacobian(y) }

  dispose() {
    if (this._bijector.dispose) this._bijector.dispose()
  }
}
