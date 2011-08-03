/*
   FALCON - The Falcon Programming Language.
   FILE: testmod.cpp

   Test of a module
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
   }
   
   virtual ~TestFunc2() {}
   
   virtual void invoke( Falcon::VMContext* ctx, int )
   {
      static Falcon::Symbol* printlSym = 
            ctx->vm()->modSpace()->findExportedSymbol( "printl" );
      
      fassert( printlSym != 0 && printlSym->value(ctx)->isFunction() );
      
      ctx->pushData("Hello from a function using printl!");
      ctx->call( printlSym->value(ctx)->asFunction(), 1 );
   }
};

FALCON_MODULE_DECL 
{
   Falcon::Module* mod = new Falcon::Module("testmod");
   
   mod->addFunction( "testFunc", &TestFunc, true );
   
   (*mod) << new TestFunc2;
   
   return mod;
}

/* end of testmod.cpp */
