/*
   FALCON - The Falcon Programming Language.
   FILE: stmtglobal.cpp

   Statatement -- global
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 27 Jan 2013 18:13:51 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/stmtglobal.cpp"

#include <falcon/psteps/stmtglobal.h>

#include <falcon/trace.h>
#include <falcon/expression.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/textwriter.h>

#include <falcon/symbol.h>
#include <falcon/function.h>
#include <falcon/module.h>

#include <falcon/stderrors.h>

#include <falcon/engine.h>
#include <falcon/synclasses.h>

#include <vector>

namespace Falcon
{

class StmtGlobal::Private
{
public:
   typedef std::vector<const Symbol*> SymbolVector;
   SymbolVector m_symbols;

   Private() {}
   ~Private() {
      SymbolVector::iterator viter = m_symbols.begin();
      while( viter != m_symbols.end() ) {
         (*viter)->decref();
         ++viter;
      }
   }

};

StmtGlobal::StmtGlobal( int32 line, int32 chr ):
         Statement(line, chr)
{
   FALCON_DECLARE_SYN_CLASS(stmt_global)
   _p = new Private;
   apply = apply_;
}

StmtGlobal::StmtGlobal( const StmtGlobal& other ):
         Statement(other)
{
   FALCON_DECLARE_SYN_CLASS(stmt_global)

   _p = new Private;
   apply = apply_;
   
   Private::SymbolVector::iterator viter = other._p->m_symbols.begin();
   Private::SymbolVector::iterator vend = other._p->m_symbols.end();
   while( viter != vend )
   {
      (*viter)->incref();
      _p->m_symbols.push_back(*viter);
      ++viter;
   }
}

StmtGlobal::~StmtGlobal()
{
   delete _p;
}

void StmtGlobal::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );
   if( _p->m_symbols.empty() )
   {
      tw->write( "/* Blank StmtGlobal */" );
   }
   else
   {
      tw->write("global ");
      Private::SymbolVector::iterator viter = _p->m_symbols.begin();
      Private::SymbolVector::iterator vend = _p->m_symbols.end();
      bool bDone = false;
      while( viter != vend )
      {
         const Symbol* sym = *viter;
         if( ! bDone ) {
            bDone = true;
         }
         else {
            tw->write(", ");
         }
         tw->write(sym->name());
         ++viter;
      }
   }

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}

bool StmtGlobal::addSymbol( const String& name )
{
   if( alreadyAdded(name) ) {
      return false;
   }

   // resolve as local
   const Symbol* sym = Engine::getSymbol(name);
   _p->m_symbols.push_back(sym);
   return true;
}

bool StmtGlobal::addSymbol( const Symbol* var )
{
   if( alreadyAdded(var->name()) ) {
      return false;
   }
   
   _p->m_symbols.push_back(var);
   return true;
}


bool StmtGlobal::alreadyAdded( const String& name ) const
{
   Private::SymbolVector::iterator viter = _p->m_symbols.begin();
   Private::SymbolVector::iterator vend = _p->m_symbols.end();
   while( viter != vend )
   {
     const Symbol* sym = *viter;
     if( sym->name() == name ) {
        return true;
     }

     ++viter;
   }

   return false;
}

void StmtGlobal::store( DataWriter* stream ) const
{
   uint32 size = _p->m_symbols.size();
   stream->write(size);

   Private::SymbolVector::iterator viter = _p->m_symbols.begin();
   Private::SymbolVector::iterator vend = _p->m_symbols.end();
   while( viter != vend )
   {
     const Symbol* sym = *viter;
     stream->write( sym->name() );

     ++viter;
   }
}

void StmtGlobal::restore( DataReader* stream )
{
   String name;

   uint32 size;
   stream->read(size);
   for(uint32 i = 0; i < size; ++i )
   {
      // reuse the same memory for performance
      name.size(0);
      stream->read(name);
      // resolve as local
      const Symbol* sym = Engine::getSymbol( name );
      _p->m_symbols.push_back( sym );
   }
}
   

void StmtGlobal::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtGlobal* self = static_cast<const StmtGlobal*>(ps);
   TRACE( "StmtGlobal::apply -- %s", self->describe().c_ize() );
   Private::SymbolVector& symbols = self->_p->m_symbols;
   fassert( ! symbols.empty() );
   
   Private::SymbolVector::iterator viter = symbols.begin();
   Private::SymbolVector::iterator vend = symbols.end();
   while( viter != vend )
   {
      const Symbol* sym = *viter;
      // Ignore the fact that the symbol is (probably) local and resolve it global.
      Item* data = ctx->resolveGlobal( sym, false );

      if( data == 0 )
      {
         Function* current = ctx->currentFrame().m_function;
         String none;
         const String* name = current->module() != 0 ?
                     &current->module()->name() : &none;

         // the symbol should be defined as global in the current module.
         ctx->raiseError(
                  new CodeError( ErrorParam(e_undef_sym, self->line(), *name )
                           .symbol( current->name() )
                           .extra( sym->name() ) ) );
         return;
      }

      ctx->defineSymbol(sym, data);
      ++viter;
   }

   // we're out of businss...
   ctx->popCode();
   // add the result of the expression.
   ctx->pushData(Item());
}

}

/* end of stmtglobal.cpp */
