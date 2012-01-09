/*
   FALCON - The Falcon Programming Language.
   FILE: testmod.cpp

   Test of a module
 
   Try with this script:

   import from testmod
   testFunc()
   testFunc2()
   testFunc3()

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 03 Aug 2011 21:58:46 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/falcon.h>

#include <iostream>

/** A traditional VM-based EXT function.
 */
void TestFunc( Falcon::VMachine* )
{
   std::cout << "Hello world from a module!" << std::endl;
}

/** A new Function -- directly invoking printl.
 
 This function will try to find printl in the global exports the first time
 it gets loaded.
 
 */
class TestFunc2: public Falcon::Function
{
public:
   TestFunc2():
      Function("testFunc2")
   {
      //addParam(" nothing ")
      m_funcPrintl = 0;
      
   }
   
   virtual ~TestFunc2() {}
   
   virtual void invoke( Falcon::VMContext* ctx, int )
   {
      // each module has a different entity, and a module cannot be linked within a different vm.
      if( m_funcPrintl == 0 )
      {
         Falcon::Symbol* printlSym = 
            ctx->vm()->modSpace()->findExportedSymbol( "printl" );
         fassert( printlSym != 0 && printlSym->getValue(0)->isFunction() );
         
         m_funcPrintl = printlSym->getValue(0)->asFunction();
      }      
      
      ctx->pushData("Hello from a function using printl!");
      ctx->call( m_funcPrintl , 1 );
   }
   
private:
   Falcon::Function* m_funcPrintl;
};


/** A function with static linkage.
 
 This function uses a pointer to the printl function that gets resolved
 during linking via the addImport request.
 */
class TestFunc3: public Falcon::Function
{
public:
   Falcon::Function* m_funcPrintl;
   TestFunc3():
      Function("testFunc3")
   {
      m_funcPrintl = 0;
      
   }
   
   virtual ~TestFunc3() {}
   
   virtual void invoke( Falcon::VMContext* ctx, int )
   {
      if( m_funcPrintl != 0 )
      {
         ctx->pushData(
            "Hello from a function using printl -- with static linkage!!!");
         ctx->call( m_funcPrintl , 1 );
      }
   }
};


/** A module with extended capabilities.
 
 It is not strictly necessary to extend the Module class,
 as all the features are available to the base class and
 virtuality is not explicitly involved.
 
 However, using an extended module helps to keep together some
 features, as i.e. the callback function in response to link resolution events.
 */
class TestModule: public Falcon::Module
{
public:
   /** We keep a pointer to an item that we need we should talk with. */
   TestFunc3* m_TheTestFunc3;   
   
   /** This callback is invoked when the "printl" function is found.*/
   static Falcon::Error* onPrintlResolved( Falcon::Module* requester, Falcon::Module* , Falcon::Symbol* sym )
   {   
      // printl should really be a function in a global symbol ,but...
      if( sym->defaultValue() == 0 || ! sym->defaultValue()->isFunction() )
      {
         return new Falcon::LinkError( Falcon::ErrorParam( 
               Falcon::e_link_error, __LINE__, requester->name() )
            .extra( "printl is not a global function!" ) );
      }

      // We know the requester is an instance of our module.
      static_cast<TestModule*>(requester)->m_TheTestFunc3->m_funcPrintl = 
                                                      sym->defaultValue()->asFunction();

      // we have no error to signal. 
      return 0;
   }

   TestModule():
      Module( "TestModule" )
   {
      // save a place where we can access it.
      m_TheTestFunc3 = new TestFunc3;
      
      (*this)
            << new TestFunc2
            << m_TheTestFunc3;
      
      // add a traditional function.
      addFunction( "testFunc", &TestFunc, true );
      
      // try to add static resolution.
      // by default, symbols are searched in the exported symbol table.
      addImportRequest( &onPrintlResolved, "printl" );
   }
   
   virtual ~TestModule();
};


TestModule::~TestModule()
{
   // Nothing to do.
}


FALCON_MODULE_DECL 
{
   Falcon::Module* mod = new TestModule;
   return mod;
}

/* end of testmod.cpp */
