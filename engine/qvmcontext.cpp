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
#include <falcon/dyncompiler.h>

namespace Falcon
{

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
	QVMContext* wctx = static_cast<QVMContext*>( ctx );

	ctx->popCode();  // not really necessary, but...

	// save the result -- and remove it from stack.
	Item tmp = wctx->topData();

	if( !tmp.isNil() )
		wctx->m_result.append( tmp );

	wctx->popData();

	if( wctx->m_items.length() > 0 )
	{
		Item itm = wctx->m_items.at( 0 );
		wctx->m_items.remove(0,1);

		wctx->pushCode( wctx->m_stepComplete );

		if( itm.isFunction() )
			wctx->call( itm.asFunction() );

		else
			wctx->callItem( itm );
	}

	else
		wctx->m_evtComplete->set();
}

class QVMContext::PStepErrorGate: public SynTree
{
	public:
		PStepErrorGate()
		{
			apply = apply_;
		}
		virtual ~PStepErrorGate() {}

		void describeTo( String& tgt, int ) const
		{
			tgt = "QVMContext::PStepErrorGate";
		}

	private:
		static void apply_( const PStep* ps, VMContext* ctx );
};

void QVMContext::PStepErrorGate::apply_( const PStep*, VMContext* ctx )
{
	MESSAGE( "QVMContext::PStepErrorGate::apply_" );

	QVMContext* wctx = static_cast<QVMContext*>( ctx );

	ctx->popCode(); // not really necessary, the ctx is going to die, but...

	if( ctx->thrownError() == 0 )
	{
		CodeError* error = FALCON_SIGN_ERROR( CodeError, e_uncaught );
		error->raised( ctx->raised() );
		wctx->completeWithError( error );
		error->decref();
	}
	else
	{
		wctx->completeWithError( ctx->thrownError() );
	}
}

//============================================================================
// Main QVMContext
//============================================================================

QVMContext::QVMContext( Process* prc, ContextGroup* grp ):
	VMContext( prc, grp ),
	m_completeCbFunc( 0 ),
	m_completeData( 0 ),
	m_completionError( 0 )
{
	// event is hand-reset.
	m_evtComplete = new Event( false, false );

	// create the completion step
	m_stepComplete = new PStepComplete;

	// prepare the error gate.
	StmtTry* errorGate = new StmtTry;
	errorGate->catchSelect().append( new PStepErrorGate );
	m_stepErrorGate = errorGate;

	m_baseFrame = new SynFunc( "<base>" );
}

QVMContext::~QVMContext()
{
	if( m_completionError != 0 )
	{
		m_completionError->decref();
	}

	delete m_evtComplete;
	delete m_stepComplete;
	delete m_baseFrame;
}

void QVMContext::onComplete()
{
	if( m_completeCbFunc != 0 )
	{
		m_completeCbFunc( this, m_completeData );
	}
}

void QVMContext::start( Function* f, int32 np, Item const* params )
{
    ItemArray itm;

    for( int32 i = 0; i < np; ++i )
        itm.append( params[i] );

	if( m_items.empty() )
	{
		m_evtComplete->reset();
		reset();
		m_items.append( Item(f) );
		process()->startContext( this );
	}

	else
		m_items.append( Item(f) );
}

//void QVMContext::start( Closure* closure, int32 np, Item const* params )
//{
////	call( closure, np, params );
////	process()->startContext( this );
//}

void QVMContext::startItem( const Item& item, int32 np, Item const* params )
{
    ItemArray itm;

    for( int32 i = 0; i < np; ++i )
        itm.append( params[i] );

	if( m_items.empty() )
	{
		m_evtComplete->reset();
		reset();
		m_items.append( item );
		process()->startContext( this );
	}

	else
		m_items.append( item );
}

void QVMContext::setOnComplete( complete_cbfunc func, void* data )
{
	m_completeCbFunc = func;
	m_completeData = data;
}

void QVMContext::gcPerformMark()
{
	VMContext::gcPerformMark();
	m_result.gcMark( m_currentMark );
}

bool QVMContext::wait( int32 to ) const
{
	bool result = m_evtComplete->wait( to );
	if( m_completionError != 0 )
	{
		throw m_completionError;
	}
	return result;
}

void QVMContext::completeWithError( Error* error )
{
	error->incref();

	if( m_completionError != 0 )
	{
		m_completionError->decref();
	}

	m_completionError = error;

	swapOut();
}

void QVMContext::reset()
{
	VMContext::reset();
	if( m_completionError != 0 )
	{
		m_completionError->decref();
		m_completionError = 0;
	}

	call( m_baseFrame );
	pushCodeWithUnrollPoint( m_stepErrorGate );
	pushCode( m_stepComplete );
}

}

/* end of QVMContext.cpp */
