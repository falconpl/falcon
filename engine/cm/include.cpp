/*
   FALCON - The Falcon Programming Language.
   FILE: include.h

   Falcon core module -- Dynamic compile function
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 05 Feb 2013 17:34:06 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/include.cpp"

#include <falcon/cm/include.h>
#include <falcon/cm/textreader.h>
#include <falcon/cm/textstream.h>
#include <falcon/classes/classstream.h>

#include <falcon/vm.h>
#include <falcon/modspace.h>
#include <falcon/vmcontext.h>
#include <falcon/module.h>
#include <falcon/syntree.h>
#include <falcon/error.h>
#include <falcon/errors/paramerror.h>
#include <falcon/itemdict.h>
#include <falcon/modloader.h>
#include <falcon/stdsteps.h>
#include <falcon/stdhandlers.h>
#include <falcon/stdhandlers.h>
#include <falcon/classes/classmodule.h>



namespace Falcon {
namespace Ext {

// The constructor sets up our secondary PStep and the parameters.

Function_include::Function_include():
         Function("include"),
         m_stepModLoaded( this )
{
   parseDescription("file:S, inputEnc:[S], path:[S], symDict:[D]" );
}


Function_include::~Function_include()
{}


// This is the entry point in the function
void Function_include::invoke( VMContext* ctx , int32 /* paramCount */)
{
   static Engine* engine = Engine::instance();
   TRACE( "Function_include::invoke -- Called from %d(%p) in %d(%p)",
            ctx->id(), ctx, ctx->process()->id(), ctx->process() )

   // Get all the parametrs. It's a waste if paramCount is < 4,
   // but the code below becomes simpler.
   Item *i_file = ctx->param(0);
   Item *i_enc = ctx->param(1);
   Item *i_path = ctx->param(2);
   Item *i_syms = ctx->param(3);

   // Do a simple check
   // Theoretically, the parameter mask set in the constructor could be used
   // for parsing, but I prefer to parse them manually as it's always
   // more efficient.
   if( i_file == 0 || ! i_file->isString()
      || (i_syms != 0 && ! (i_syms->isDict() || i_syms->isNil())  )
      || (i_enc != 0 && !(i_enc->isString() || i_enc->isNil()) )
      || (i_path != 0 && !(i_path->isString() || i_path->isNil()) )
      )
   {
      // were in Function_include::invoke method,
      // and the class know how to report a parameter mismatch error.
      throw this->paramError(__LINE__, SRC );
   }

   String& file = *i_file->asString();

   // Get the vm module loader and space.
   ModSpace* masterMS = ctx->process()->modSpace();
   ModLoader* masterLoader = masterMS->modLoader();

   // check if the given encoding is plausible.
   const String* encoding = 0;
   if( i_enc != 0 && i_enc->isString() )  // might be given but nil...
   {
      if( engine->getTranscoder(*i_enc->asString()) )
      {
         encoding = i_enc->asString();
      }
      else {
         throw FALCON_SIGN_XERROR( ParamError, e_param_range,
                  .extra("Invalid encoding" +*i_enc->asString() ));
      }
   }

   if( encoding == 0 )
   {
      // use the default encoding
      encoding = &masterLoader->sourceEncoding();
   }

   // same for path
   const String* path = (i_path == 0 || i_path->isNil() ) ?
            &masterLoader->getSearchPath() :
            i_path->asString();


   ModSpace* childMS = new ModSpace(ctx->process(), masterMS );
   // configure the child module loader
   ModLoader* childLoader = childMS->modLoader();
   childLoader->sourceEncoding( *encoding );
   childLoader->setSearchPath( *path );

   // we don't keep any extra reference, so we can give the module space
   // to the garbage collector.
   // The first data pushed is accessible as ctx->local(0)
   ctx->pushData( FALCON_GC_HANDLE(childMS) );

   // prepare to abandon the frame
   ctx->pushCode( &m_stepModLoaded );

   TRACE1( "Function_include::invoke -- Loading \"%s\" with path \"%s\" encoding \"%s\"",
            file.c_ize(), path->c_ize(), encoding->c_ize() );

   // this might throw on I/O error, or go deep to load more modules
   Module* callingModule = ctx->callDepth() > 1 ? ctx->callerFrame(1).m_function->module() : 0;
   childMS->loadModuleInContext( file, true, false, true, ctx, callingModule );
}


void Function_include::PStepModLoaded::apply_(const PStep* pstep, VMContext* ctx )
{
   // We're in a PStep of our function...
   const Function_include::PStepModLoaded* self = static_cast<const Function_include::PStepModLoaded*>(pstep);
   // ... to get our function back, we need the the pstep.
   Function_include* func = self->m_owner;

   // Ok, we have this local structure:
   // local(0) --> childMS
   // local(1) --> the loaded module, put there by childMS->loadModuleInContext
   // Parameters are still where we left them.
   Item* i_syms = ctx->param(3);
   Item* i_module = ctx->local(1);

   // check that the data type of the item is REALLY what we want.
   // (at least as a debug assert)
   fassert( i_module->asClass() == Engine::handlers()->moduleClass() );
   Module* module = static_cast<Module*>(i_module->asInst());

   // Now we must decide if we have to run the module or just take
   // -- some symbols out of it.

   // see our current status in cf.m_seqId
   CodeFrame& cf = ctx->currentCode();

   //it's useful to have some trace for debug
   TRACE( "Function_include::PStepModLoaded::apply -- step %d/1", cf.m_seqId );

   switch( cf.m_seqId )
   {
   case 0:
      cf.m_seqId = 1; // prepare for next stage
      // we must check if we need to execute main or not.
      if( i_syms != 0 && i_syms->isDict() )
      {
         // the user doesn't want us to call main.
         func->getModSymbols( *i_syms->asDict(), module );
      }
      else {
         // has the module a __main__ to run?
         Function* theMain = module->getMainFunction();
         if( theMain != 0 )
         {
            // great, let's call it;
            ctx->call(theMain);
            // we're ready to be called back, with seqId == 1, when done.
            return;
         }
      }

      /* no break */
   }

   // ready to return the module;
   // but better to copy it in the C stack while the VM stack
   // gets unrolled

   Item copy = *i_module;
   ctx->returnFrame( copy );

   // notice that since we unroll the frame, we need not to pop this PStep.
}

// Utility function
void Function_include::getModSymbols( ItemDict& syms, Module *module )
{

   // dictionaries are traversed by the means of an enumerator.

   class Rator: public ItemDict::Enumerator
   {
   public:
      Rator( Module* src ): m_module(src) {}
      virtual ~Rator() {}

      virtual void operator()( const Item& key, Item& value )
      {
         if( key.isString() )
         {
            // search the mantra (Invocable entity: object, class or function)
            String* mantraName = key.asString();
            Mantra* mantra = m_module->getMantra(*mantraName);
            if( mantra != 0 )
            {
               // if we have it, return it to the caller by setting the value
               value.setUser( mantra->handler(), mantra );

               // Notice that we use the GC handler of the mantra, but we don't
               // give it to the GC for automatic clearing. The mantra is owned by
               // the module, and will be destroyed when the module is destroyed.
               // However, the GC will mark the mantra, and this will keep its
               // module alive as long as the mantra is somewhere around.
            }
         }
      }

   private:
      Module* m_module;
   };

   // perform the enumeration.
   Rator rator(module);
   syms.enumerate( rator );
}

}
}

/* end of include.cpp */
