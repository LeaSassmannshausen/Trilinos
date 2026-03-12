// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef __Teko_LU2x2TriangularStrategy_hpp__
#define __Teko_LU2x2TriangularStrategy_hpp__

#include "Teko_LU2x2Strategy.hpp"

namespace Teko {

/** @brief Strategy for LU2x2 preconditioning that uses an externally supplied
 *         Schur complement operator.
 *
 * This strategy builds \f$A_{00}^{-1}\f$ using a configured inverse factory and
 * obtains the Schur operator from the request handler (or falls back to
 * \f$A_{11}\f$ if configured).
 */
class LU2x2TriangularStrategy : public LU2x2Strategy {
 public:
  //! default Constructor
  LU2x2TriangularStrategy();

  //! Constructor to set the inverse factories.
  LU2x2TriangularStrategy(const Teuchos::RCP<InverseFactory>& invFA,
                          const Teuchos::RCP<InverseFactory>& invS);

  //! Destructor (does nothing)
  virtual ~LU2x2TriangularStrategy() {}

  /** returns the first (approximate) inverse of \f$A_{00}\f$ */
  virtual const Teko::LinearOp getHatInvA00(const Teko::BlockedLinearOp& A,
                                            BlockPreconditionerState& state) const;

  /** returns the second (approximate) inverse of \f$A_{00}\f$ */
  virtual const Teko::LinearOp getTildeInvA00(const Teko::BlockedLinearOp& A,
                                              BlockPreconditionerState& state) const;

  /** returns an (approximate) inverse of externally supplied Schur operator */
  virtual const Teko::LinearOp getInvS(const Teko::BlockedLinearOp& A,
                                       BlockPreconditionerState& state) const;

  /** \brief Build the internals of the state from a parameter list.
   *
   * Supported settings:
   *  - "Inverse Type"
   *  - "Inverse A00 Type"
   *  - "Inverse Schur Type"
   *  - "External Schur Name" (default: "External Schur Complement")
   *  - "Use External Schur" (default: true)
   *  - "External Schur Is Already Inverse" (default: false)
   */
  virtual void initializeFromParameterList(const Teuchos::ParameterList& settings,
                                           const InverseLibrary& invLib);

 protected:
  /** Initialize cached inverse operators in the state.
   */
  void initializeState(const Teko::BlockedLinearOp& A, BlockPreconditionerState& state) const;

  // how to invert the matrices
  Teuchos::RCP<InverseFactory> invFactoryA00_;
  Teuchos::RCP<InverseFactory> invFactoryS_;

  std::string schurRequestName_;
  bool useExternalSchur_;
  bool externalSchurIsAlreadyInverse_;
};

}  // end namespace Teko

#endif
