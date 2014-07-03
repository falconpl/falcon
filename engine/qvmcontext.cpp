/*
   FALCON - The Falcon Programming Language.
   FILE: QVMContext.cpp

   Falcon virtual machine -- waitable VM context.
   -------------------------------------------------------------------
   Author: AlexRou

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/qvmcontext.cpp"

#include <falcon/qvmcontext.h>
#include <falcon/pstep.h>
#include <falcon/syntree.h>
#include <falcon/psteps/stmttry.h>
#include <falcon/stderrors.h>
#include <falcon/synfunc.h>
#include <falcon/stream.h>

namespace Falcon
{

class QVMContext::QueueItem
{
	public:
		String m_name;
		Item m_itm;
		int32 m_nParams;
		Item const* m_params;
		PStep* m_onComplete;
		PStep* m_onError;

		QueueItem( String name, Item itm, int32 nParams = 0, Item const* params = 0, PStep* onComplete = 0, PStep* onError = 0 ):
			m_name( name ), m_itm( itm ), m_nParams( nParams ), m_params( params ), m_onComplete( onComplete ), m_onError( onError )
		{

		}
};

class QVMContext::PStepComplete: public PStep
{
	public:
		PStepComplete()
		{
			apply = apply_;
		}
		virtual ~PStepComplete() {}

	private:
		static void apply_( const PStep* ps, VMContext* ctx );
};

void QVMContext::PStepComplete::apply_( const PStep*, VMContext* ctx )
{
	MESSAGE( "QVMContext::PStepComplete::apply_" );

	ctx->popCode();
	ctx->popCode();

	ctx->popData();
}

class QVMContext::PStepErrorGate: public SynTree
{
	public:
		PStepErrorGate()
		{
			apply = apply_;
		}
		virtual ~PStepErrorGate() {}

		void describeTo( String& tgt ) const
		{
			tgt = "QVMContext::PStepErrorGate";
		}

	private:
		static void apply_( const PStep* ps, VMContext* ctx );
};

void QVMContext::PStepErrorGate::apply_( const PStep*, VMContext* ctx )
{
	MESSAGE( "QVMContext::PStepErrorGate::apply_" );

	//QVMContext* qctx = static_cast<QVMContext*>( ctx );

	ctx->popCode();
	ctx->popCode();

	if( ctx->thrownError() == 0 )
	{
		CodeError* error = FALCON_SIGN_ERROR( CodeError, e_uncaught );
		error->raised( ctx->raised() );
		String str = error->describe();
		ctx->process()->stdErr()->write( str.c_ize(), str.length() );
		error->decref();
	}
	else
	{
		String str = ctx->thrownError()->describe();
		ctx->process()->stdErr()->write( str.c_ize(), str.length() );
	}
}

class QVMContext::PStepDeque: public PStep
{
	public:
		PStepDeque()
		{
			apply = apply_;

			StmtTry* errorGate = new StmtTry;
			errorGate->catchSelect().append( new PStepErrorGate );
			m_stepErrorGate = errorGate;

			m_stepComplete = new PStepComplete;
		}
		virtual ~PStepDeque() {}

	private:
		PStep* m_stepErrorGate;
		PStep* m_stepComplete;

		static void apply_( const PStep* ps, VMContext* ctx );
};

void QVMContext::PStepDeque::apply_( const PStep* ps, VMContext* ctx )
{
	QVMContext* qctx = static_cast<QVMContext*>( ctx );
	const PStepDeque* psd = static_cast<const PStepDeque*>( ps );

	MESSAGE( "QVMContext::PStepDeque::apply_" );

	if( qctx->m_Queue )
	{
		QVMContext::Node* tmp = qctx->m_Queue;
		QVMContext::QueueItem* itm = tmp->data;
		qctx->m_Queue = tmp->next;
		delete tmp;

		qctx->pushCodeWithUnrollPoint( itm->m_onError ? itm->m_onError : psd->m_stepErrorGate );
		qctx->pushCode( itm->m_onComplete ? itm->m_onComplete : psd->m_stepComplete );
		qctx->callItem( itm->m_itm, itm->m_nParams, itm->m_params );
		qctx->m_running = itm->m_name;
		qctx->m_isSleeping = false;
		delete[] itm->m_params;
	}

	else
	{
		if( qctx->events() == VMContext::evtTerminate )
			qctx->swapOut();

		else
		{
		    qctx->m_running = "Sleeping";
			qctx->m_isSleeping = true;
			qctx->m_QueueEnd = 0;
		}
	}
}

//============================================================================
// Main QVMContext
//============================================================================

QVMContext::QVMContext( Process* prc, ContextGroup* grp ):
	VMContext( prc, grp ),
	m_Queue( 0 ),
	m_QueueEnd( 0 ),
	m_running( "Sleeping" )
{
	// create the completion step
	m_stepQueue = new PStepDeque;

	m_baseFrame = new SynFunc( "<base>" );

	m_isSleeping = true;

	call( m_baseFrame );
	pushCode( m_stepQueue );
	process()->startContext( this );
}

QVMContext::~QVMContext()
{
	delete m_stepQueue;
	delete m_baseFrame;

	while( m_Queue )
	{
		Node* tmp = m_Queue->next;
		delete m_Queue;
		m_Queue = tmp;
	}
}

void QVMContext::start( String name, Item item, int32 np, Item const* params, PStep* complete, PStep* error )
{
	m_isSleeping = false;

	Node* tmp = new Node;
	tmp->data = new QueueItem( name, item, np, params, complete, error );
	tmp->next = 0;

	if( m_Queue && m_QueueEnd )
		m_QueueEnd->next = tmp;

	else
		m_Queue = tmp;

	m_QueueEnd = tmp;
}

void QVMContext::gcPerformMark()
{
	VMContext::gcPerformMark();
}

void QVMContext::onComplete()
{

}

}

/* end of QVMContext.cpp */
