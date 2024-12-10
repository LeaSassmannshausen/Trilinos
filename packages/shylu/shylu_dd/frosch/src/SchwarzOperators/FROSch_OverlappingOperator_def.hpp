// @HEADER
// *****************************************************************************
//               ShyLU: Scalable Hybrid LU Preconditioner and Solver
//
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_OVERLAPPINGOPERATOR_DEF_HPP
#define _FROSCH_OVERLAPPINGOPERATOR_DEF_HPP

#include <FROSch_OverlappingOperator_decl.hpp>


namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC,class LO,class GO,class NO>
    OverlappingOperator<SC,LO,GO,NO>::OverlappingOperator(ConstXMatrixPtr k,
                                                          ParameterListPtr parameterList) :
    SchwarzOperator<SC,LO,GO,NO> (k,parameterList)
    {
        FROSCH_DETAILTIMER_START_LEVELID(overlappingOperatorTime,"OverlappingOperator::OverlappingOperator");
        if (!this->ParameterList_->get("Combine Values in Overlap","Restricted").compare("Averaging")) {
            Combine_ = Averaging;
        } else if (!this->ParameterList_->get("Combine Values in Overlap","Restricted").compare("Full")) {
            Combine_ = Full;
        } else if (!this->ParameterList_->get("Combine Values in Overlap","Restricted").compare("Restricted")) {
            Combine_ = Restricted;
        }
	if (this->ParameterList_->get("Use Pressure Correction",false)) {
            FROSCH_NOTIFICATION("FROSch::Overlapping Operator",(this->Verbose_) && this->ParameterList_->get("Use Local Pressure Correction",false),"Use local projections to correct pressure.");
            FROSCH_NOTIFICATION("FROSch::Overlapping Operator",(this->Verbose_) && this->ParameterList_->get("Use Global Pressure Correction",false),"Use global projection to correct pressure.");

            //FROSCH_NOTIFICATION("FROSch::Overlapping Operator",(this->Verbose_),"Use pressure projection to correct pressure:: Local Pressure Correction");<<<<<<<<<<<<s

            this->aProjection_ = ExtractPtrFromParameterList<XMultiVector >(*this->ParameterList_,"Projection");    
        }
    }

    template <class SC,class LO,class GO,class NO>
    OverlappingOperator<SC,LO,GO,NO>::~OverlappingOperator()
    {
        SubdomainSolver_.reset();
    }

 // Y = alpha * A^mode * X + beta * Y                                                                                                                                                                                            
    template <class SC,class LO,class GO,class NO>
    void OverlappingOperator<SC,LO,GO,NO>::apply(const XMultiVector &x,
                                                 XMultiVector &y,
                                                 bool usePreconditionerOnly,
                                                 ETransp mode,
                                                 SC alpha,
                                                 SC beta) const
    {


        FROSCH_TIMER_START_LEVELID(applyTime,"OverlappingOperator::apply");

        FROSCH_ASSERT(this->IsComputed_,"FROSch::OverlappingOperator: OverlappingOperator has to be computed before calling apply()");
        if (XTmp_.is_null() || XTmp_->getNumVectors() != x.getNumVectors()) {
            XTmp_ = MultiVectorFactory<SC,LO,GO,NO>::Build(x.getMap(),x.getNumVectors());
        }
        *XTmp_ = x;
        if (!usePreconditionerOnly && mode == NO_TRANS) {
            this->K_->apply(x,*XTmp_,mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
        }
        // AH 11/28/2018: For Epetra, XOverlap_ will only have a view to the values of XOverlapTmp_. Therefore, xOverlapTmp should not be deleted before XOverlap_ is used.
        if (YOverlap_.is_null() || YOverlap_->getNumVectors() != x.getNumVectors()) {
            YOverlap_ = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMatrix_->getDomainMap(),x.getNumVectors());
        } else {
            YOverlap_->replaceMap(OverlappingMatrix_->getDomainMap());
        }
        
        
        {
        FROSCH_TIMER_START_LEVELID(applyTime,"Beginning of Apply 2" );

          // AH 11/28/2018: replaceMap does not update the GlobalNumRows. Therefore, we have to create a new MultiVector on the serial Communicator. In Epetra, we can prevent to copy the MultiVector.                             
          if (XTmp_->getMap()->lib() == UseEpetra) {
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
            if (XOverlapTmp_.is_null() || XOverlap_->getNumVectors() != x.getNumVectors()) {
                XOverlapTmp_ = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,x.getNumVectors());
            }
            XOverlapTmp_->doImport(*XTmp_,*Scatter_,INSERT);
            const RCP<const EpetraMultiVectorT<GO,NO> > xEpetraMultiVectorXOverlapTmp = rcp_dynamic_cast<const EpetraMultiVectorT<GO,NO> >(XOverlapTmp_);
            RCP<Epetra_MultiVector> epetraMultiVectorXOverlapTmp = xEpetraMultiVectorXOverlapTmp->getEpetra_MultiVector();
            const RCP<const EpetraMapT<GO,NO> >& xEpetraMap = rcp_dynamic_cast<const EpetraMapT<GO,NO> >(OverlappingMatrix_->getRangeMap());
            Epetra_BlockMap epetraMap = xEpetraMap->getEpetra_BlockMap();
            double *A;
            int MyLDA;
            epetraMultiVectorXOverlapTmp->ExtractView(&A,&MyLDA);
            RCP<Epetra_MultiVector> epetraMultiVectorXOverlap(new Epetra_MultiVector(::View,epetraMap,A,MyLDA,x.getNumVectors()));
            XOverlap_ = RCP<EpetraMultiVectorT<GO,NO> >(new EpetraMultiVectorT<GO,NO>(epetraMultiVectorXOverlap));
#else
            FROSCH_ASSERT(false,"HAVE_XPETRA_EPETRA not defined.");
#endif
          } else {
            if (XOverlap_.is_null()) {
              XOverlap_ = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,x.getNumVectors());
            } else {
                XOverlap_->replaceMap(OverlappingMap_);
            }
            XOverlap_->doImport(*XTmp_,*Scatter_,INSERT);
            XOverlap_->replaceMap(OverlappingMatrix_->getRangeMap());
          }

         // END TIMER                                                                                                                                                                                                               
        // This one is timed                                                                                                                                                                                                        
        //this->MpiComm_->barrier();
        //this->MpiComm_->barrier();
        //this->MpiComm_->barrier();
        //this->MpiComm_->barrier();
        }
        {
          FROSCH_TIMER_START_LEVELID(applyTime,"Call to subdomain solver apply");

          SubdomainSolver_->apply(*XOverlap_,*YOverlap_,mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());

          {
            FROSCH_TIMER_START_LEVELID(applyTime,"Collecting rest of time in barriers()");
            //this->MpiComm_->barrier();
            //this->MpiComm_->barrier();
            //this->MpiComm_->barrier();
            //this->MpiComm_->barrier();
          }
        }// End TIMER                                                                                                                                                                 
        {
          FROSCH_TIMER_START_LEVELID(applyTime,"raplaceMap");

          YOverlap_->replaceMap(OverlappingMap_);

        }
	    // Is it necessary to apply the projection here locally or can we apply it a the end to the global solution. Probably this will be done in the sum operator
        // Does restricted Schwarz any influence on the pressure
        // Reading aTmp from paramterfile
	//  std::cout << " Use local pressure correction " << this->ParameterList_->get("Use Local Pressure Correction",false) << " a projection is null: " << this->aProjection_.is_null() << std::endl; 
        if (!this->aProjection_.is_null() && (this->ParameterList_->get("Use Local Pressure Correction",false) == true)){

            FROSCH_TIMER_START_LEVELID(applyTime,"Apply Pressure Projection");

            RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout));
            XMultiVectorPtr a = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMatrix_->getDomainMap(),x.getNumVectors());
            // Distribute it with overlap based on overlapping matrix
            a->doImport(*this->aProjection_,*Scatter_,INSERT);
            a->replaceMap(OverlappingMap_);
	   
            //SCVecPtr a_values = a->getDataNonConst(0);  
            //SCVecPtr y_values = YOverlap_->getDataNonConst(0);
            XMultiVectorConstPtr aConst = a;    

            //double sumAY = 0.;
            Teuchos::Array<SC> sumAY(1);

            {
                FROSCH_TIMER_START_LEVELID(applyTime,"Sum up AY");
                // for(int i=0; i< a_values.size(); i++)
                //     sumAY += a_values[i] * y_values[i];
                YOverlap_->dot(*aConst,sumAY);
            }

            // double sumAA = 0.;
            Teuchos::Array<SC> sumAA(1);
            {
                FROSCH_TIMER_START_LEVELID(applyTime,"Sum up AA");

                // for(int i=0; i< a_values.size(); i++)
                //     sumAA += a_values[i] * a_values[i];
                a->dot(*aConst,sumAA);
            }
            double aint = 1./sumAA[0];
            SC scaling = aint*sumAY[0]; 

            //YOverlap_->update(-scaling,*aConst,1);
            //YOverlap_->describe(*fancy,VERB_EXTREME);
            {              
                FROSCH_TIMER_START_LEVELID(applyTime,"Apply Scaling");
                // for(int i=0; i< a->getDataNonConst(0).size(); i++)
                //     y_values[i] -= scaling*  a_values[i];
                YOverlap_->update(-scaling,*aConst,1);
            }
            
            // Sanity Check
            //Teuchos::Array<SC> ortho(1);
            //YOverlap_->dot(*aConst,ortho);
            //if(abs(ortho[0]) >= 1.e-12 )
            //    cout << " ########### ORTHO CHECK on proc " << YOverlap_->getMap()->getComm()->getRank() << "= "  << ortho[0] << " ############ " << endl;
        }
        XTmp_->putScalar(ScalarTraits<SC>::zero());
         // END TIMER                                                                                                                                                                 
        
        ConstXMapPtr yMap;
          ConstXMapPtr yOverlapMap;
        {
          FROSCH_TIMER_START_LEVELID(applyTime,"getMap()");
           yMap = y.getMap();
           yOverlapMap = YOverlap_->getMap();

        // END TIMER                                                                                                                                                                  
        /*this->MpiComm_->barrier();
        this->MpiComm_->barrier();
        this->MpiComm_->barrier();
        this->MpiComm_->barrier();*/
        }
        {
          FROSCH_TIMER_START_LEVELID(applyTime,"Combine Overlap mode=restricted");

    if (Combine_ == Restricted) {
#if defined(HAVE_XPETRA_KOKKOS_REFACTOR) && defined(HAVE_XPETRA_TPETRA)
            if (XTmp_->getMap()->lib() == UseTpetra) {
                auto yLocalMap = yMap->getLocalMap();
                auto yLocalOverlapMap = yOverlapMap->getLocalMap();
                // run local restriction on execution space defined by local-map                                                                                                      
                using XMap            = typename SchwarzOperator<SC,LO,GO,NO>::XMap;
                using execution_space = typename XMap::local_map_type::execution_space;
                Kokkos::RangePolicy<execution_space> policy (0, yMap->getLocalNumElements());

                using xTMVector    = Xpetra::TpetraMultiVector<SC,LO,GO,NO>;
                // Xpetra wrapper for Tpetra MV                                                                                                                                       
                auto yXTpetraMVector = rcp_dynamic_cast<const xTMVector>(YOverlap_, true);
                auto xXTpetraMVector = rcp_dynamic_cast<      xTMVector>(XTmp_, true);
                // Tpetra MV                                                                                                                                                          
                auto yTpetraMVector = yXTpetraMVector->getTpetra_MultiVector();
                auto xTpetraMVector = xXTpetraMVector->getTpetra_MultiVector();
                // View                                                                                                                                                               
                auto yView = yTpetraMVector->getLocalViewDevice(Tpetra::Access::ReadOnly);
                auto xView = xTpetraMVector->getLocalViewDevice(Tpetra::Access::ReadWrite);
                for (UN j=0; j<y.getNumVectors(); j++) {
                    Kokkos::parallel_for(
                      "FROSch_OverlappingOperator::applyLocalRestriction", policy,
                      KOKKOS_LAMBDA(const int i) {
                        GO gID = yLocalMap.getGlobalElement(i);
                        LO lID = yLocalOverlapMap.getLocalElement(gID);
                        xView(i, j) = yView(lID, j);
                      });
                }
                Kokkos::fence();
            } else
#endif
            {
                GO globID = 0;
                LO localID = 0;
                for (UN i=0; i<y.getNumVectors(); i++) {
                    ConstSCVecPtr yOverlapData_i = YOverlap_->getData(i);
                    for (UN j=0; j<yMap->getLocalNumElements(); j++) {
                        globID = yMap->getGlobalElement(j);
                        localID = yOverlapMap->getLocalElement(globID);
                        XTmp_->getDataNonConst(i)[j] = yOverlapData_i[localID];
                    }
                }
            }
        }
        else {
            XTmp_->doExport(*YOverlap_,*Scatter_,ADD);
        }
    /*this->MpiComm_->barrier();
        this->MpiComm_->barrier();
        this->MpiComm_->barrier();
        this->MpiComm_->barrier();*/

        }// END TIMER
	{
          FROSCH_TIMER_START_LEVELID(applyTime,"Combine Overlap mode=Averaging");


          if (Combine_ == Averaging) {
            ConstSCVecPtr scaling = Multiplicity_->getData(0);
            for (UN j=0; j<XTmp_->getNumVectors(); j++) {
              SCVecPtr values = XTmp_->getDataNonConst(j);
              for (UN i=0; i<values.size(); i++) {
                values[i] = values[i] / scaling[i];
              }
            }
          }
	  /*          this->MpiComm_->barrier();
          this->MpiComm_->barrier();
          this->MpiComm_->barrier();
          this->MpiComm_->barrier();*/

        }// END TIMER                                                                                                                                                                 

        { FROSCH_TIMER_START_LEVELID(applyTime,"if(!usePreconditionerOnly && mode != NO_TRANS) K_->apply()");
          if (!usePreconditionerOnly && mode != NO_TRANS) {
            this->K_->apply(*XTmp_,*XTmp_,mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
          }
	  /*this->MpiComm_->barrier();
        this->MpiComm_->barrier();
        this->MpiComm_->barrier();
        this->MpiComm_->barrier();*/

        } // END TIMER                                                                                                                                                                

	    // We could use the global approach of the projection and apply it here! Should have same convergence and same number of iterations
        // if (!this->aProjection_.is_null() && (this->ParameterList_->get("Use Global Pressure Correction",false) == true)){

        //     RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout));
        //     //XMultiVectorPtr aGlobal = MultiVectorFactory<SC,LO,GO,NO>::Build(x.getMap(),x.getNumVectors());
        //     // Distribute it with overlap based on overlapping matrix
        //     //aGlobal->doImport(*this->aProjection_,*Scatter_,INSERT);
        //     //a->replaceMap(OverlappingMap_);
        //     //a->describe(*fancy,VERB_EXTREME);
        //     //YOverlap_->describe(*fancy,VERB_EXTREME);
        //     // Define constant MVs for dot operations
        //     XMultiVectorConstPtr XTmpConst = XTmp_;
        //     XMultiVectorConstPtr aConst = this->aProjection_;    

        //     // compute a*y 
        //     Teuchos::Array<SC> sumAY(1);
        //     this->aProjection_->dot(*XTmpConst,sumAY);
        //     // compute (a^T*a)^-1
        //     Teuchos::Array<SC> sumAA(1);
        //     this->aProjection_->dot(*aConst,sumAA);
        //     double aint = 1./sumAA[0];
        //     SC scaling = aint*sumAY[0]; // scaling for a vector : I * y - scaling * a , with scaling = (a^T*a)^-1 * a * y 
        //     //cout << " Processor " << YOverlap_->getMap()->getComm()->getRank() << " SumAA " << sumAA << " sumAY " << sumAY << " aInt " << aint << " scaling " << scaling << endl;
        //     //YOverlap_->describe(*fancy,VERB_EXTREME);
        //     XTmp_->update(-scaling,*aConst,1);
        //     //YOverlap_->describe(*fancy,VERB_EXTREME);

        //     // Sanity Check
        //     Teuchos::Array<SC> ortho(1);
        //     XTmp_->dot(*aConst,ortho);
        //     if(abs(ortho[0]) >= 1.e-12 )
        //         cout << " ########### ORTHO CHECK on proc " << YOverlap_->getMap()->getComm()->getRank() << "= "  << ortho[0] << " ############ " << endl;
        // }

	
        {
          FROSCH_TIMER_START_LEVELID(applyTime,"y.update()");
        y.update(alpha,*XTmp_,beta);

        /*this->MpiComm_->barrier();
        this->MpiComm_->barrier();
        this->MpiComm_->barrier();
        this->MpiComm_->barrier();*/

        } // END TIMER                                                                                                                                                                
	/* this->MpiComm_->barrier();
              this->MpiComm_->barrier();
              this->MpiComm_->barrier();
              this->MpiComm_->barrier();*/

    }

    template <class SC,class LO,class GO,class NO>
    int OverlappingOperator<SC,LO,GO,NO>::initializeOverlappingOperator()
    {
        FROSCH_DETAILTIMER_START_LEVELID(initializeOverlappingOperatorTime,"OverlappingOperator::initializeOverlappingOperator");
        Scatter_ = ImportFactory<LO,GO,NO>::Build(this->getDomainMap(),OverlappingMap_);
        if (Combine_ == Averaging) {
            Multiplicity_ = MultiVectorFactory<SC,LO,GO,NO>::Build(this->getRangeMap(),1);
            XMultiVectorPtr multiplicityRepeated;
            multiplicityRepeated = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,1);
            multiplicityRepeated->putScalar(ScalarTraits<SC>::one());
            XExportPtr multiplicityExporter = ExportFactory<LO,GO,NO>::Build(multiplicityRepeated->getMap(),this->getRangeMap());
            Multiplicity_->doExport(*multiplicityRepeated,*multiplicityExporter,ADD);
        }
        return 0; // RETURN VALUE
    }

    /*template <class SC,class LO,class GO,class NO>
    int OverlappingOperator<SC,LO,GO,NO>::initializeSubdomainSolver(ConstXMatrixPtr localMat)
    {
        FROSCH_DETAILTIMER_START_LEVELID(initializeSubdomainSolverTime,"OverlappingOperator::initializeSubdomainSolver");
        SubdomainSolver_ = SolverFactory<SC,LO,GO,NO>::Build(localMat,
                                                             sublist(this->ParameterList_,"Solver"),
                                                             string("Solver (Level ") + to_string(this->LevelID_) + string(")"));
        SubdomainSolver_->initialize();
        return 0; // RETURN VALUE
    }*/

    template <class SC,class LO,class GO,class NO>
    int OverlappingOperator<SC,LO,GO,NO>::computeOverlappingOperator()
    {
        FROSCH_DETAILTIMER_START_LEVELID(computeOverlappingOperatorTime,"OverlappingOperator::computeOverlappingOperator");

        updateLocalOverlappingMatrices();
        bool reuseSymbolicFactorization = this->ParameterList_->get("Reuse: Symbolic Factorization",true);
        
        if (!this->IsComputed_) {
            reuseSymbolicFactorization = false;
        }

        if (!reuseSymbolicFactorization) {
            if (this->IsComputed_ && this->Verbose_) cout << "FROSch::OverlappingOperator : Recomputing the Symbolic Factorization" << endl;
            SubdomainSolver_ = SolverFactory<SC,LO,GO,NO>::Build(OverlappingMatrix_,
                                                                 sublist(this->ParameterList_,"Solver"),
                                                                 string("Solver (Level ") + to_string(this->LevelID_) + string(")"));
            SubdomainSolver_->initialize();
        } 
        else {
            FROSCH_ASSERT(!SubdomainSolver_.is_null(),"FROSch::OverlappingOperator: SubdomainSolver_.is_null()");
            SubdomainSolver_->updateMatrix(OverlappingMatrix_,true);
        }
        
        this->IsComputed_ = true;
        return SubdomainSolver_->compute();
        /*if (!reuseSymbolicFactorization || SubdomainSolver_.is_null()) {
            // initializeSubdomainSolver is called during symbolic only if reuseSymbolicFactorization=true
            // so if reuseSymbolicFactorization=false, we always call initializeSubdomainSolver 
            if (this->IsComputed_ && this->Verbose_) cout << "FROSch::OverlappingOperator : Recomputing the Symbolic Factorization" << endl;
            initializeSubdomainSolver(this->OverlappingMatrix_);
        } else if (this->IsComputed_) {
            // if !IsComputed, then this is the first timing calling "compute" after initializeSubdomainSolver is called in symbolic phase
            // so no need to do anything
            SubdomainSolver_->updateMatrix(this->OverlappingMatrix_,true);
        }
        this->IsComputed_ = true;
        return SubdomainSolver_->compute();*/
    }
}

#endif
