/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __itkPSMEntropyModelFilter_hxx
#define __itkPSMEntropyModelFilter_hxx
#include "itkPSMEntropyModelFilter.h"

#include "itkZeroCrossingImageFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{

template <class TImage, class TShapeMatrix>
PSMEntropyModelFilter<TImage, TShapeMatrix>::PSMEntropyModelFilter()
{
    // Default parameters for the multiscale functionality
    m_CurrentScale   = 0;
    this->SetNumberOfScales(1);

    m_Initializing = false;

    m_ParticleSystem = PSMParticleSystem<Dimension>::New();

    // Here we create the various necessary components for the Shape
    // Model optimization. First we construct the Shape Matrix.
    m_ShapeMatrix = ShapeMatrixType::New();
    m_ShapeMatrix->SetDomainsPerShape(1);

    // Now we create the Particle Entropy Function. This is the entropy
    // estimation of particle position distributions on shape surfaces.
    m_ParticleEntropyFunction
            = PSMParticleEntropyFunction<typename ImageType::PixelType, Dimension>::New();

    // Set the number of domains per shape for the particle entropy function
    m_ParticleEntropyFunction->SetDomainsPerShape(1);

    // Now initialize the Shape Space entropy function and point it to
    // the Shape Matrix attribute.  The Shape Space entropy function
    // computes entropy of the shape samples in n-dimensional shape
    // space.
    m_ShapeEntropyFunction = PSMShapeEntropyFunction<Dimension>::New();
    m_ShapeEntropyFunction->SetShapeMatrix(m_ShapeMatrix);

    // Now allocate an optimizer and set iteration callback.
    m_Optimizer = OptimizerType::New();
    m_IterateCallback  = itk::MemberCommand<PSMEntropyModelFilter<TImage, TShapeMatrix> >::New();
    m_IterateCallback->SetCallbackFunction(this, &PSMEntropyModelFilter<TImage, TShapeMatrix>::OptimizerIterateCallback);
    m_Optimizer->AddObserver(itk::IterationEvent(), m_IterateCallback);

    // Finally, we need a cost function that combines the
    // ParticleEntropy function with the ShapeEntropy function.
    m_CostFunction = PSMTwoCostFunction<Dimension>::New();
    m_CostFunction->SetFunctionA(m_ParticleEntropyFunction);
    m_CostFunction->SetFunctionB(m_ShapeEntropyFunction);

    m_Initialized = false;

    // Register the Shape Matrix as an attribute of the particle system.
    // This ensures that the Shape Matrix will receive any particle
    // AddPosition/RemovePosition/UpdatePosition events, so that it can
    // update its data accordingly.
    m_ParticleSystem->RegisterAttribute(m_ShapeMatrix);
}


template<class TImage, class TShapeMatrix>
void
PSMEntropyModelFilter<TImage, TShapeMatrix>::AllocateDataCaches()
{
    // Set up the various data caches that the optimization functions will use.
    // Sigma cache is caching a parameter for the local particle entropy
    // computation that is updated for each particle.
    m_SigmaCache = PSMContainerArrayAttribute<double, Dimension>::New();
    m_ParticleSystem->RegisterAttribute(m_SigmaCache);
    m_ParticleEntropyFunction->SetSpatialSigmaCache(m_SigmaCache);

    // m_MeanCurvatureCache = PSMMeanCurvatureAttribute<typename ImageType::PixelType, Dimension>::New();
    // m_ParticleSystem->RegisterAttribute(m_MeanCurvatureCache);
}

template <class TImage, class TShapeMatrix>
void
PSMEntropyModelFilter<TImage, TShapeMatrix>::AllocateWorkingImages()
{
    m_WorkingImages.resize(this->GetNumberOfInputs());
    for (unsigned int i = 0; i < this->GetNumberOfInputs(); i++)
    {
        m_WorkingImages[i] = const_cast<TImage *>(this->GetInput(i));
    }
}

template <class TImage, class TShapeMatrix>
void
PSMEntropyModelFilter<TImage, TShapeMatrix>::AllocateDomainsAndNeighborhoods()
{
    // Allocate all the necessary domains and neighborhoods. This must be done
    // *after* registering the attributes to the particle system since some of
    // them respond to AddDomain.
    for (unsigned int i = 0; i < this->GetNumberOfInputs(); i++)
    {
        m_DomainList.push_back( PSMImplicitSurfaceDomain<typename
                                ImageType::PixelType, Dimension>::New() );
        m_NeighborhoodList.push_back( PSMSurfaceNeighborhood<ImageType>::New() );
        m_DomainList[i]->SetSigma(m_WorkingImages[i]->GetSpacing()[0] * 2.0);
        m_DomainList[i]->SetImage(m_WorkingImages[i]);

        m_ParticleSystem->AddDomain(m_DomainList[i]);
        m_ParticleSystem->SetNeighborhood(i, m_NeighborhoodList[i]);
    }
}

template <class TImage, class TShapeMatrix>
void
PSMEntropyModelFilter<TImage, TShapeMatrix>::InitializeCorrespondences()
{
    // If the user has specified correspondence points on the input,
    // then make sure the number of lists of correspondence points
    // matches the number of inputs (distance transforms).  Everything
    // should match this->GetNumberOfInputs().
    if (m_InputCorrespondencePoints.size() > 0)
    {
        if (this->GetNumberOfInputs() != m_InputCorrespondencePoints.size() )
        {
            itkExceptionMacro("The number of inputs does not match the number of correspondence point lists.");
        }
        else
        {
            for (unsigned int i = 0; i < m_InputCorrespondencePoints.size(); i++)
            {
                this->GetParticleSystem()->AddPositionList(m_InputCorrespondencePoints[i],i);
            }
        }
    }
    else // No input correspondences are specified, so add a point to each surface.
    {
        this->CreateSingleCorrespondence();
    }

    // Push position information out to all observers (necessary to
    // correctly fill out the shape matrix)
    this->GetParticleSystem()->SynchronizePositions();
}

template <class TImage, class TShapeMatrix>
void
PSMEntropyModelFilter<TImage, TShapeMatrix>::GenerateData()
{
    // First, undo the default InPlaceImageFilter functionality that will
    // release our inputs.
    this->SetInPlace(false);

    // If this is the first call to GenerateData, then we need to allocate some structures
    if (this->GetInitialized() == false)
    {
        this->AllocateWorkingImages();
        this->AllocateDataCaches();
        this->GetOptimizer()->SetCostFunction(m_CostFunction);

        this->AllocateDomainsAndNeighborhoods();

        // Point the optimizer to the particle system.
        this->GetOptimizer()->SetParticleSystem(this->GetParticleSystem());
        //      this->ReadTransforms();
        this->InitializeCorrespondences();
        this->InitializeOptimizationFunctions();

        this->SetInitialized(true);
    }

    // This filter can be run in "initialization mode", which causes all
    // data structures to allocate and initialize, but does not run the
    // actual optimization.
    if (this->GetInitializing() == true) return;

    // Multiscale optimization.  This can be stopped and restarted by
    // calling GenerateData again by maintaining m_CurrentScale.
    for (unsigned int scale = m_CurrentScale; scale < m_NumberOfScales; scale++)
    {
        m_CurrentScale = scale;

        // Set up the optimization parameters for this scale, unless
        // this is a restart from a previous call to GenerateData. This
        // section deals with convergence criteria.
        if (this->GetAbortGenerateData() == false)
        {
            // This section deals with convergence.
            m_Optimizer->SetTolerance(m_Tolerances[scale]);
            m_Optimizer->SetMaximumNumberOfIterations(m_MaximumNumberOfIterations[scale]);

            // Check whether an auto initial regularization mode is required,
            // if so, ignore the input initial regularization and compute the regularization
            // term according to the current status of the shape matrix
            if(m_RegularizationInitialMode == "auto")
            {
                m_RegularizationInitial[scale] = this->ComputeShapeMatrixRegularizationTerm();

                // the final regularization is set to keep a certain ratio between final and initial
                m_RegularizationFinal[scale] = m_RegularizationInitial[scale] * 0.1f;
            }

            // Set up the exponentially-decreasing regularlization constant.  If
            // the decay span is greater than 1 iteration, then we will set up
            // the annealing approach.  Otherwise, the optimizer will simply use
            // its constant annealing parameter.
            if (m_RegularizationDecaySpan[scale] >= 1.0f)
            {
                m_ShapeEntropyFunction->SetMinimumVarianceDecay(m_RegularizationInitial[scale],
                                                                m_RegularizationFinal[scale],
                                                                m_RegularizationDecaySpan[scale]);
            }
            else
            {
                m_ShapeEntropyFunction->SetMinimumVariance(m_RegularizationInitial[scale]);
                m_ShapeEntropyFunction->SetHoldMinimumVariance(true);
            }
        }

        this->SetAbortGenerateData(false); // reset the abort flag

        // Finally, run the optimization.  Abort flag is checked in the
        // iteration callback registered with the optimizer. See
        // this->OptimizerIterateCallback
        this->GetOptimizer()->StartOptimization();

        // If this is multiscale, split particles for the next iteration
        // -- but not if this is the last iteration.
        if ((m_NumberOfScales > 1) && (scale != m_NumberOfScales-1) )
        {
            m_ParticleSystem->SplitAllParticles(this->GetInput()->GetSpacing()[0] * 1.0);

            // a new scale/split, so the pairwise distance between particles could change
            // set a negative values to the particle's global sigma in case
            // using the cotan pairwise potential
            if(m_ParticleEntropyFunction->GetPairwisePotentialType() == "cotan")
                m_ParticleEntropyFunction->SetGlobalSigma(-1.0f);
        }

    }
}

template <class TImage, class TShapeMatrix>
void
PSMEntropyModelFilter<TImage, TShapeMatrix>::InitializeOptimizationFunctions()
{
    // Set the minimum neighborhood radius and maximum sigma based on
    // the domain of the 1st input image.
    unsigned int maxdim = 0;
    double maxradius = 0.0;
    double spacing = this->GetInput()->GetSpacing()[0];
    for (unsigned int i = 0; i < TImage::ImageDimension; i++)
    {
        if (this->GetInput()->GetRequestedRegion().GetSize()[i] > maxdim)
        {
            maxdim = this->GetInput()->GetRequestedRegion().GetSize()[i];
            maxradius = maxdim * this->GetInput()->GetSpacing()[i];
        }
    }

    // Initialize member variables of the optimization functions.
    //  m_ParticleEntropyFunction->SetMinimumNeighborhoodRadius(maxradius / 3.0);
    m_ParticleEntropyFunction->SetMinimumNeighborhoodRadius(spacing * 5.0);
    m_ParticleEntropyFunction->SetMaximumNeighborhoodRadius(maxradius);

    m_ShapeMatrix->Initialize();
}

template <class TImage, class TShapeMatrix>
void
PSMEntropyModelFilter<TImage, TShapeMatrix>
::OptimizerIterateCallback(itk::Object *, const itk::EventObject &)
{
    // Update any optimization values based on potential changes from
    // the user during the last iteration here.  This allows for changes in a
    // thread-safe manner.

    // Now invoke any iteration events registered with this filter.
    this->InvokeEvent(itk::IterationEvent());

    // Request to terminate this filter.
    if (this->GetAbortGenerateData() == true)
    {
        m_Optimizer->StopOptimization();
    }
}

template <class TImage, class TShapeMatrix>
bool
PSMEntropyModelFilter<TImage, TShapeMatrix>
::CreateSingleCorrespondence()
{
    bool ok = true;
    for (unsigned int i = 0; i < m_ParticleSystem->GetNumberOfDomains(); i++)
    {
        typename itk::ZeroCrossingImageFilter<ImageType, ImageType>::Pointer zc =
                itk::ZeroCrossingImageFilter<ImageType, ImageType>::New();

        zc->SetInput( dynamic_cast<PSMImplicitSurfaceDomain<typename ImageType::PixelType, Dimension> *>(
                          m_ParticleSystem->GetDomain(i))->GetImage());
        zc->Update();
        itk::ImageRegionConstIteratorWithIndex<ImageType> it(zc->GetOutput(),
                                                             zc->GetOutput()->GetRequestedRegion());

        bool done = false;
        for (it.GoToReverseBegin(); !it.IsAtReverseEnd() && done == false; --it)
        {
            if (it.Get() == 1.0)
            {
                PointType pos;
                dynamic_cast<PSMImplicitSurfaceDomain<typename ImageType::PixelType, Dimension> *>(
                            m_ParticleSystem->GetDomain(i))->GetImage()->TransformIndexToPhysicalPoint(it.GetIndex(), pos);
                try
                {
                    m_ParticleSystem->AddPosition(pos, i);
                    done = true;
                }
                catch(itk::ExceptionObject &)
                {
                    done = false;
                    ok = false;
                }
            }
        }
    }
    return ok;
}

template <class TImage, class TShapeMatrix>
double
PSMEntropyModelFilter<TImage, TShapeMatrix>
::ComputeShapeMatrixRegularizationTerm()
{
    double RegularizationTerm = 0.0;

    // NOTE: This code requires that indices be contiguous, i.e. it wont work if
    // you start deleting particles.
    const unsigned int num_samples = m_ShapeMatrix->cols();
    const unsigned int num_dims    = m_ShapeMatrix->rows();

    typedef vnl_vector<double> vnl_vector_type;
    typedef vnl_matrix<double> vnl_matrix_type;

    vnl_matrix_type points_minus_mean(num_dims, num_samples);
    vnl_vector_type means(num_dims);

    // Compute the covariance matrix.
    // (A is D' in Davies paper)
    // Compute the mean shape vector.
    for (unsigned int j = 0; j < num_dims; j++)
    {
        double total = 0.0;
        for (unsigned int i = 0; i < num_samples; i++)
        {
            total += m_ShapeMatrix->operator()(j, i);
        }
        means(j) = total/(double)num_samples;
    }


    for (unsigned int j = 0; j < num_dims; j++)
    {
        for (unsigned int i = 0; i < num_samples; i++)
        {
            points_minus_mean(j, i) = m_ShapeMatrix->operator()(j, i) - means(j);
        }
    }

    vnl_matrix_type A =  points_minus_mean.transpose()
            * points_minus_mean * (1.0/((double)(num_samples-1+1e-10)));


    vnl_svd<double> svdA(A);

    // note that for symmetric matrices, singular values and eigen values coincide
    const double singularThreshold = 1.0e-6;
    RegularizationTerm             = fabs(svdA.sigma_min()) + singularThreshold;

    double sum    = 0.0;
    for (unsigned int i = 0; i < num_samples; i++)
        sum += (fabs(svdA.W(i,i)) * double(fabs(svdA.W(i,i)) > singularThreshold) );

    double cumsum = 0.0;
    unsigned int i;
    for(i = 0 ; i < num_samples; i++)
    {
        cumsum += fabs(svdA.W(i,i));
        if((cumsum/sum) >= 0.97)
            break;
    }

    for(unsigned int ip = i+1; ip < num_samples; ip++)
        RegularizationTerm += (fabs(svdA.W(ip,ip)) * double(fabs(svdA.W(ip,ip)) > singularThreshold));

    return RegularizationTerm;
}

} // end namespace

#endif
