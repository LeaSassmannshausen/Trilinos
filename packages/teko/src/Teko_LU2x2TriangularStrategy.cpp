// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Teko_LU2x2TriangularStrategy.hpp"

namespace Teko {

LU2x2TriangularStrategy::LU2x2TriangularStrategy()
    : schurRequestName_("External Schur Complement"),
      useExternalSchur_(true),
      externalSchurIsAlreadyInverse_(false) {}

LU2x2TriangularStrategy::LU2x2TriangularStrategy(const Teuchos::RCP<InverseFactory>& invFA,
                                                 const Teuchos::RCP<InverseFactory>& invS)
    : invFactoryA00_(invFA),
      invFactoryS_(invS),
      schurRequestName_("External Schur Complement"),
      useExternalSchur_(true),
      externalSchurIsAlreadyInverse_(false) {}

const Teko::LinearOp LU2x2TriangularStrategy::getHatInvA00(const Teko::BlockedLinearOp& A,
                                                           BlockPreconditionerState& state) const {
  initializeState(A, state);
  return state.getModifiableOp("invA00");
}

const Teko::LinearOp LU2x2TriangularStrategy::getTildeInvA00(
    const Teko::BlockedLinearOp& A, BlockPreconditionerState& state) const {
  initializeState(A, state);
  return state.getModifiableOp("invA00");
}

const Teko::LinearOp LU2x2TriangularStrategy::getInvS(const Teko::BlockedLinearOp& A,
                                                      BlockPreconditionerState& state) const {
  initializeState(A, state);
  return state.getModifiableOp("invS");
}

void LU2x2TriangularStrategy::initializeState(const Teko::BlockedLinearOp& A,
                                              BlockPreconditionerState& state) const {
  Teko_DEBUG_SCOPE("LU2x2TriangularStrategy::initializeState", 10);

  // no work to be done
  if (state.isInitialized()) return;

  TEUCHOS_ASSERT(invFactoryA00_ != Teuchos::null);
  TEUCHOS_ASSERT(invFactoryS_ != Teuchos::null);

  // extract sub blocks
  LinearOp A00 = Teko::getBlock(0, 0, A);
  LinearOp B = Teko::getBlock(0, 1, A);                                              
  // build inverse A00
  {
    Teko_DEBUG_SCOPE("Building inverse(A00)", 5);
    ModifiableLinearOp& invA00 = state.getModifiableOp("invA00");
    if (invA00 == Teuchos::null)
      invA00 = buildInverse(*invFactoryA00_, A00);
    else
      rebuildInverse(*invFactoryA00_, A00, invA00);
  }

  // fetch/build inverse Schur operator
  {
    Teko_DEBUG_SCOPE("Building inverse(S)", 5);

    LinearOp schurOp = Teuchos::null;
    if (useExternalSchur_) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          getRequestHandler() == Teuchos::null, std::logic_error,
          "LU2x2TriangularStrategy requires a request handler for the external Schur operator.");
      schurOp = getRequestHandler()->request<Teko::LinearOp>(Teko::RequestMesg(schurRequestName_));
      TEUCHOS_TEST_FOR_EXCEPTION(
          schurOp == Teuchos::null, std::logic_error,
          "LU2x2TriangularStrategy callback \"" + schurRequestName_ + "\" returned null.");
    } else {
      schurOp = Teko::getBlock(1, 1, A);
    }

    ModifiableLinearOp& invS = state.getModifiableOp("invS");
    if (externalSchurIsAlreadyInverse_) {
      invS = Teuchos::rcp_const_cast<Thyra::LinearOpBase<double> >(schurOp);
    } else if (invS == Teuchos::null) {
      invS = buildInverse(*invFactoryS_, schurOp);
    } else {
      rebuildInverse(*invFactoryS_, schurOp, invS);
    }
  }

  // mark state as initialized
  state.setInitialized(true);
}

void LU2x2TriangularStrategy::initializeFromParameterList(const Teuchos::ParameterList& pl,
                                                          const InverseLibrary& invLib) {
  Teko_DEBUG_SCOPE("LU2x2TriangularStrategy::initializeFromParameterList", 10);

  std::string invStr = "", invA00Str = "", invSStr = "";
#if defined(Teko_ENABLE_Amesos)
  invStr = "Amesos";
#elif defined(Teko_ENABLE_Amesos2)
  invStr = "Amesos2";
#endif

  // parse the parameter list
  if (pl.isParameter("Inverse Type")) invStr = pl.get<std::string>("Inverse Type");
  if (pl.isParameter("Inverse A00 Type")) invA00Str = pl.get<std::string>("Inverse A00 Type");
  if (pl.isParameter("Inverse Schur Type")) invSStr = pl.get<std::string>("Inverse Schur Type");
  if (pl.isParameter("External Schur Name"))
    schurRequestName_ = pl.get<std::string>("External Schur Name");
  if (pl.isParameter("Use External Schur")) useExternalSchur_ = pl.get<bool>("Use External Schur");
  if (pl.isParameter("External Schur Is Already Inverse"))
    externalSchurIsAlreadyInverse_ = pl.get<bool>("External Schur Is Already Inverse");

  // set defaults as needed
  if (invA00Str == "") invA00Str = invStr;
  if (invSStr == "") invSStr = invStr;

  // build inverse factory objects
  invFactoryA00_ = invLib.getInverseFactory(invA00Str);
  if (invA00Str == invSStr)
    invFactoryS_ = invFactoryA00_;
  else
    invFactoryS_ = invLib.getInverseFactory(invSStr);

  // pre-request the external Schur operator if needed
  if (useExternalSchur_) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        getRequestHandler() == Teuchos::null, std::logic_error,
        "LU2x2TriangularStrategy requires a request handler for the external Schur operator.");
    getRequestHandler()->preRequest<Teko::LinearOp>(Teko::RequestMesg(schurRequestName_));
  }
}

}  // end namespace Teko
