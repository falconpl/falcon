/*
   FALCON - The Falcon Programming Language
   FILE: error.cpp

   Error management.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 21:15:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/error.cpp"

#include <falcon/setup.h>
#include <falcon/error.h>
#include <falcon/item.h>
#include <falcon/tracestep.h>
#include <falcon/mt.h>
#include <falcon/sys.h>
#include <falcon/class.h>
#include <falcon/engine.h>

#include <deque>

namespace Falcon {

static const String errorDesc( int code )
{
   static String unk("Unknown error");

   switch( code )
   {
      #define FLC_MAKE_ERROR_MESSAGE_SELECTOR
      #include <falcon/error_messages.h>
   }

   return unk;
}

//==================================================
// Error
//==================================================

class Error_p
{
public:
   std::deque<TraceStep> m_steps;
   std::deque<Error*> m_subErrors;
};


Error::Error( const String& name, const ErrorParam &params ):
   m_refCount( 1 ),
   m_bHasRaised(false),
   m_name( name ),
   m_handler(0)
{
   _p = new Error_p;
   set(params);
}


Error::Error( const String& name ):
   m_refCount( 1 ),
   m_bHasRaised(false),
   m_name( name ),
   m_handler(0)
{
   _p = new Error_p;
}

Error::Error( const Class* handler ):
   m_refCount( 1 ),
   m_bHasRaised(false),
   m_name( handler->name() ),
   m_handler( handler )
{
   _p = new Error_p;
}



void Error::set( const ErrorParam& params )
{
   m_errorCode = params.m_errorCode ;
   m_description = params.m_description ;
   m_extra = params.m_extra ;
   m_mantra = params.m_symbol ;
   m_module = params.m_module ;
   m_path = params.m_path;
   m_signature = params.m_signature;
   m_line= params.m_line ;
   m_chr= params.m_chr ;
   m_sysError= params.m_sysError ;
   m_origin= params.m_origin ;
   m_catchable= params.m_catchable ;
}


Error::~Error()
{
   std::deque<Error*>::const_iterator iter = _p->m_subErrors.begin();
   while( iter != _p->m_subErrors.end() )
   {
      (*iter)->decref();
      ++iter;
   }

   delete _p;
}


void Error::incref() const
{
   atomicInc( m_refCount );
}


void Error::decref()
{
   if( atomicDec( m_refCount ) <= 0 )
   {
      delete this;
   }
}


void Error::describeTo( String &target, bool addSignature ) const
{
   heading( target );

   if ( addSignature && ! m_signature.empty() )
   {
      target += "\n   Signed by: " + m_signature;
   }

   if ( ! _p->m_steps.empty() )
   {
      target += "\n   Traceback:";

      std::deque<TraceStep>::const_iterator iter = _p->m_steps.begin();
      while( iter != _p->m_steps.end() )
      {
          target += "\n   ";
          const TraceStep& step = *iter;
          step.toString( target );
          ++iter;
      }
   }

   if (! _p->m_subErrors.empty() )
   {
      target += "\n   Because of:\n";
      std::deque<Error*>::const_iterator iter = _p->m_subErrors.begin();
      while( iter != _p->m_subErrors.end() )
      {
         target += (*iter)->describe(addSignature) +"\n";
         
         ++iter;
      }
   }

   if ( m_bHasRaised )
   {
      target += "   Raised item: " + m_raised.describe();
   }
}


String &Error::heading( String &target ) const
{
   target += handler()->name();
   target += " ";

   switch( m_origin )
   {
   case ErrorParam::e_orig_compiler: target += "CP"; break;
   case ErrorParam::e_orig_linker: target += "LK"; break;
   case ErrorParam::e_orig_loader: target += "LD"; break;
   case ErrorParam::e_orig_vm: target += "VM"; break;
   case ErrorParam::e_orig_runtime: target += "RT"; break;
   case ErrorParam::e_orig_mod: target += "MD"; break;
   case ErrorParam::e_orig_script: target += "SC"; break;
   default: target += "??"; break;
   }

   uint32 ecode = (uint32) m_errorCode;

   for ( int number = 1000; number > 0; number /= 10 )
   {
      int64 cipher = ecode / number;
      ecode %= number;
      target.writeNumber( cipher );
   }

   if ( m_sysError != 0 )
   {
      target += "(sys: ";
      target.writeNumber( (int64) m_sysError );
      String temp;
      Sys::_describeError( m_sysError, temp );
      target += " " + temp;
      target += ")";
   }

   if( m_line != 0 || m_module.size() != 0 )
      target += " at ";

   if ( m_module.size() != 0 )
   {
      target += m_module;
      if ( m_mantra.size() != 0 )
         target += "." + m_mantra;
      target += ":";
   }

   if ( m_line != 0 )
      target.writeNumber( (int64) m_line );

   if ( m_chr != 0 )
   {
      target += ":";
      target.writeNumber( (int64) m_chr );
   }

   if ( m_description.size() > 0 )
   {
      target += ": " + m_description;
   }
   else {
      target += ": " + errorDesc( m_errorCode );
   }

   if ( m_extra.size() > 0 )
   {
      target += " (" + m_extra + ")";
   }

   if ( ! m_raised.isNil() )
   {
      String temp;
      m_raised.describe( temp );
      target += "\n"+ temp;
   }

   return target;
}


void Error::addTrace( const TraceStep& tb )
{
   _p->m_steps.push_back( tb );
}


void Error::appendSubError( Error *error )
{
   error->incref();
   
   _p->m_subErrors.push_back( error );
}


void Error::scriptize( Item& tgt )
{
   incref();
   tgt.setUser( FALCON_GC_STORE( handler(), this ) );
}


const Class* Error::handler() const
{
   if ( m_handler == 0 )
   {
      m_handler = Engine::instance()->getError( m_name );
      fassert( m_handler != 0 );
   }

   return m_handler;
}


void Error::enumerateSteps( Error::StepEnumerator &rator ) const
{
   std::deque<TraceStep>::const_iterator iter = _p->m_steps.begin();
   while( iter != _p->m_steps.end() )
   {
      const TraceStep& ts = *iter;
      ++iter;
      if( ! rator( ts ) ) break;
   }
}


void Error::enumerateErrors( Error::ErrorEnumerator &rator ) const
{
   std::deque<Error*>::const_iterator iter = _p->m_subErrors.begin();
   while( iter != _p->m_subErrors.end() )
   {
      Error* error = *iter;
      ++iter;
      if( ! rator( *error ) ) break;
   }
}


Error* Error::getBoxedError() const
{
   if( _p->m_subErrors.empty() )
      return 0;
   return _p->m_subErrors.front();
}


bool Error::hasTraceback() const
{
   return ! _p->m_steps.empty();
}

bool Error::hasSubErrors() const
{
   return ! _p->m_subErrors.empty();
}


} // namespace Falcon

/* end of error.cpp */

