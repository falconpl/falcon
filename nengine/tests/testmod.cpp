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

void TestFunc( Falcon::VMachine* )
{
   std::cout << "Hello wrold from a module!" << std::endl;
}

class TestFunc2: public Falcon::Function
{
public:
   TestFunc2():
      Function("testFunc2")
   {
      setDeterm(true);
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
         fassert( printlSym != 0 && printlSym->value(0)->isFunction() );
         
         m_funcPrintl = printlSym->value(0)->asFunction();
      }      
      
      ctx->pushData("Hello from a function using printl!");
      ctx->call( m_funcPrintl , 1 );
   }
   
private:
   Falcon::Function* m_funcPrintl;
};


class TestFunc3: public Falcon::Function
{
public:
   Falcon::Function* m_funcPrintl;
   TestFunc3():
      Function("testFunc3")
   {
      setDeterm(true);
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


class TestModule: public Falcon::Module
{
public:
   TestFunc3* m_TheTestFunc3;   
   
   static Falcon::Error* onPrintlResolved( Falcon::Module* requester, Falcon::Module* , Falcon::Symbol* sym )
   {   
      // printl should really be a function in a global symbol ,but...
      if( sym->value(0) == 0 || ! sym->value(0)->isFunction() )
      {
         return new Falcon::LinkError( Falcon::ErrorParam( 
               Falcon::e_link_error, __LINE__, requester->name() )
            .extra( "printl is not a global function!" ) );
      }

      // We know the requester is an instance of our module.
      static_cast<TestModule*>(requester)->m_TheTestFunc3->m_funcPrintl = 
                                                      sym->value(0)->asFunction();

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
