/*
   FALCON - The Falcon Programming Language.
   FILE: compile.cpp

   Falcon core module -- Compile function
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 11:12:52 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/compile.cpp"

#include <falcon/cm/compile.h>
#include <falcon/cm/textreader.h>
#include <falcon/cm/textstream.h>
#include <falcon/classes/classstream.h>

#include <falcon/vm.h>
#include <falcon/stringstream.h>
#include <falcon/vmcontext.h>
#include <falcon/module.h>
#include <falcon/syntree.h>
#include <falcon/error.h>
#include <falcon/errors/genericerror.h>

#include <falcon/dyncompiler.h>

namespace Falcon {
namespace Ext {

Compile::Compile():
   Function( "compile" )
{
   signature("S|Stream|TextReader");
   addParam("code");
}

Compile::~Compile()
{
}

void Compile::invoke( VMContext* ctx , int32 params )
{
   static Class* streamClass = Engine::instance()->streamClass();
   static Class* readerClass = m_module->getClass("TextReader");
   static Class* tsClass = m_module->getClass("TextStream");

   fassert( readerClass!= 0 );
   fassert( tsClass != 0 );

   TextReader* reader = 0;
   Stream* sinput;
   bool ownReader;

   // the parameter can be a string, a stream or a text reader.
   if( params >= 1 ) {
      Item* pitem = ctx->param(0);
      Class* cls;
      void* data;

      if( pitem->isString() ) {
         sinput = new StringStream( *pitem->asString() );
         reader = new TextReader(sinput );
         ownReader = true;
      }
      else if( pitem->asClassInst(cls, data) )
      {
         if( cls->isDerivedFrom(streamClass) ) {
            StreamCarrier* sc = static_cast<StreamCarrier*>( data );
            reader = new TextReader( sc->m_underlying );
            ownReader = true;
         }
         else if( cls->isDerivedFrom( readerClass ) ) {
            TextReaderCarrier* trc = static_cast<TextReaderCarrier*>( data );
            reader = trc->m_reader;
            ownReader = false;
         }
         else if( cls->isDerivedFrom( tsClass ) ) {
            TextStreamCarrier* tsc = static_cast<TextStreamCarrier*>( data );
            reader = tsc->m_reader;
            ownReader = false;
         }
      }
   }

   if( reader == 0 ) {
      throw paramError(__LINE__, SRC );
   }

   DynCompiler dynComp( ctx );

   try
   {
      SynTree* st = dynComp.compile( reader );

      if( ownReader )
      {
         delete reader;
      }

      ctx->returnFrame( FALCON_GC_HANDLE(st) );
   }
   catch( ... )
   {
      if( ownReader )
      {
         delete reader;
      }
      throw;
   }
}

}
}

/* end of comile.cpp */
