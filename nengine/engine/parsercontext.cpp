/*
   FALCON - The Falcon Programming Language.
   FILE: compcontext.cpp

   Compilation context for Falcon source file compilation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 16 Apr 2011 18:17:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>

#include <falcon/parsercontext.h>
#include <falcon/synfunc.h>
#include <falcon/unknownsymbol.h>
#include <falcon/sourceparser.h>

#include <vector>
#include <map>

#include "falcon/globalsymbol.h"
#include "falcon/compiler.h"
#include "falcon/closedsymbol.h"
#include "falcon/expression.h"
#include "falcon/exprsym.h"
#include "falcon/error.h"


namespace Falcon {

//==================================================================
// Frame class
//

class ParserContext::CCFrame
{
   typedef union tag_elem {
      Class* cls;
      SynFunc* func;
      Statement* stmt;
      void* raw;
   } t_elem;

   typedef enum tag_type {
      t_none_type,
      t_class_type,
      t_object_type,
      t_func_type,
      t_stmt_type,
      t_base_type
   } t_type;

   CCFrame();
   CCFrame( Class* cls, bool bIsObject, GlobalSymbol* gs );
   CCFrame( SynFunc* func, GlobalSymbol* gs );
   CCFrame( Statement* stmt, SynTree* st );
   CCFrame( SynTree* st );

public:
   friend class ParserContext;

   /** Syntree element topping this frame. */
   t_elem m_elem;

   /** Type of frame */
   t_type m_type;

   /** Syntree where to add the incoming children */
   SynTree* m_branch;

   /** True if a parser state was pushed at this frame */
   bool m_bStatePushed;

   GlobalSymbol* m_sym;
};

ParserContext::CCFrame::CCFrame():
   m_type( t_none_type )
{

}

ParserContext::CCFrame::CCFrame( Class* cls, bool bIsObject, GlobalSymbol* gs ):
   m_type( bIsObject ? t_object_type : t_class_type ),
   m_branch( 0 ),
   m_bStatePushed( false ),
   m_sym(gs)
{
   m_elem.cls = cls;
}

ParserContext::CCFrame::CCFrame( SynFunc* func, GlobalSymbol* gs ):
   m_type( t_func_type ),
   m_branch( &func->syntree() ),
   m_bStatePushed( false ),
   m_sym(gs)
{
   m_elem.func = func;
}


ParserContext::CCFrame::CCFrame( Statement* stmt, SynTree* st ):
   m_type( t_stmt_type ),
   m_branch( st ),
   m_bStatePushed( false )
{
   m_elem.stmt = stmt;
}

ParserContext::CCFrame::CCFrame( SynTree* st ):
   m_type( t_base_type ),
   m_branch( st ),
   m_bStatePushed( false )
{
   m_elem.raw = 0;
}


//==================================================================
// Main parser context class
//

class ParserContext::Private
{
private:
   friend class ParserContext;

   typedef std::map< String, UnknownSymbol* > SymbolMap;

   /** Map of unknown symbols created during the current statement parsing. */
   SymbolMap m_unknown;

   typedef std::vector<ParserContext::CCFrame> FrameVector;
   FrameVector m_frames;

   Private() {}
   ~Private() {}
};


ParserContext::ParserContext( SourceParser* sp ):
   m_parser(sp),
   m_st(0),
   m_cstatement(0),
   m_cfunc(0),
   m_cclass(0),
   m_symtab(0)
{
   sp->setContext(this);
   _p = new Private;
}

ParserContext::~ParserContext()
{
   delete _p;
}

void ParserContext::openMain( SynTree* st )
{
   _p->m_frames.push_back( CCFrame( st ) );
   m_st = st;
}

void ParserContext::onStatePushed()
{
   fassert( ! _p->m_frames.empty() );
   _p->m_frames.back().m_bStatePushed = true;
}


Symbol* ParserContext::addVariable( const String& variable )
{
   TRACE("ParserContext::addVariable %s", variable.c_ize() );

   UnknownSymbol* us;
   Private::SymbolMap::const_iterator iter = _p->m_unknown.find(variable);
   if( iter == _p->m_unknown.end() )
   {
      us = new UnknownSymbol(variable);
      _p->m_unknown[variable] = us;
   }
   else
   {
      us = iter->second;
   }

   return us;
}


void ParserContext::defineSymbols( Expression* expr )
{
   TRACE("ParserContext::defineSymbols on (: %s :)", expr->describe().c_ize() );
   
   if( expr->type() == Expression::t_symbol )
   {
      Symbol* uks = static_cast<ExprSymbol*>(expr)->symbol();
      defineSymbol( uks );
   }

   //TODO: Else, to the same for each symbol in case of symbol lists.
   //TODO: Else, raise error for values or list of values.
}

void ParserContext::defineSymbol( Symbol* uks )
{
   if( uks->type() == Symbol::t_unknown_symbol )
   {
      TRACE1("ParserContext::defineSymbol trying to define symbol \"%s\"", uks->name().c_ize() );
      Symbol* nuks = findSymbol(uks->name());
      // Found?
      if( nuks == 0 )
      {
         // --no ? create it
         if( m_symtab == 0 )
         {
            // we're in the global context.
            bool bAlready;
            nuks = onGlobalDefined( uks->name(), bAlready );
         }
         else
         {
            // add it in the current symbol table.
            nuks = m_symtab->addLocal( uks->name() );
         }
      }

      // created?
      if( nuks != 0 )
      {
         TRACE("ParserContext::defineSymbol defined \"%s\" as %d",
               nuks->name().c_ize(), nuks->type() );
         // remove from the unknowns
         _p->m_unknown.erase( uks->name() );
         static_cast<UnknownSymbol*>(uks)->define(nuks);
         delete uks;
      }
      else
      {
         //TODO: Use a more fitting error code for this?
         m_parser->addError(e_undef_sym, m_parser->currentSource(), uks->declaredAt(),0, 0, uks->name() );
         
         TRACE("ParserContext::defineSymbol cannot define \"%s\"",
               uks->name().c_ize() );
         // we cannot create it; delegate to subclasses
         onUnknownSymbol( static_cast<UnknownSymbol*>(uks) );
      }
   }
   else
   {
      // else, the symbol is already ok.
      TRACE2("ParserContext::defineSymbol \"%s\" already of type %d",
               uks->name().c_ize(), uks->type() );
   }
}


bool ParserContext::checkSymbols()
{
   bool isOk = true;
   Private::SymbolMap& unknown = _p->m_unknown;

   // try with all the undefined symbols.
   TRACE("ParserContext::checkSymbols on %d syms", (int) unknown.size() );
   Private::SymbolMap::iterator iter = unknown.begin();
   while( iter != unknown.end() )
   {
      UnknownSymbol* sym = iter->second;
      Symbol* new_sym = findSymbol( sym->name() );

      if ( new_sym == 0 )
      {
         // see if it's a global or extern for our sub-class...
         TRACE1("ParserContext::checkSymbols \"%s\" is undefined, up-notifying", sym->name().c_ize() );
         new_sym = onUndefinedSymbol( sym->name() );

         // still undefined? -- we surrender, and hope that the subclass has properly raised
         if ( new_sym == 0 )
         {
            TRACE1("ParserContext::checkSymbols \"%s\" leaving this symbol undefined", sym->name().c_ize() );
         }
      }

      if( new_sym != 0 )
      {
         TRACE1("ParserContext::checkSymbols \"%s\" is now of type %d", sym->name().c_ize(), new_sym->type() );
         sym->define(new_sym);
         delete sym;
      }
      else
      {
         // we know that the symbol is lost
         m_parser->addError(e_undef_sym, m_parser->currentSource(), sym->declaredAt(),0, 0, sym->name() );
         isOk = false;
         
         // -- see if the callee wants to do something about that
         TRACE1("ParserContext::checkSymbols cannot define \"%s\"",
                  sym->name().c_ize() );
         onUnknownSymbol( sym );

         // if this parser is interactive, the calling item is discarded,
         // as such the symbol is dead.
         if( m_parser->interactive() )
         {
            delete sym;
         }
      }
      ++iter;
   }

   unknown.clear();
   return isOk;
}


Symbol* ParserContext::findSymbol( const String& name )
{
   TRACE1("ParserContext::findSymbol \"%s\"", name.c_ize() );
   if( m_symtab == 0 )
   {
      return 0;
   }

   // found at first shot?
   Symbol* sym = m_symtab->findSymbol( name );
   if( sym !=  0 )
   {
      TRACE1("ParserContext::findSymbol \"%s\" found locally", sym->name().c_ize() );
      return sym;
   }

   Private::FrameVector::const_reverse_iterator iter = _p->m_frames.rbegin();
   while( iter != _p->m_frames.rend() )
   {
      const CCFrame& frame = *iter;
      
      // skip symbol table of classes, they can't be referenced by inner stuff.
      if( frame.m_type != CCFrame::t_func_type || &frame.m_elem.func->symbols() == m_symtab )
      {
         ++iter;
         continue;
      }

      sym = frame.m_elem.func->symbols().findSymbol( name );
      if( sym !=  0 )
      {
         if( sym->type() == Symbol::t_local_symbol )
         {
            TRACE1("ParserContext::findSymbol \"%s\" found, need to be closed", sym->name().c_ize() );
            //TODO: Properly close symbols. -- this won't work
            ClosedSymbol* closym = new ClosedSymbol(name, Item());
            m_symtab->addSymbol(closym);
            return closym;
         }

         TRACE1("ParserContext::findSymbol \"%s\" found of type %d", sym->name().c_ize(), sym->type() );
         return sym;
      }
      ++iter;
   }
   
   return 0;
}


void ParserContext::addStatement( Statement* stmt )
{
   TRACE("ParserContext::addStatement type %d", (int) stmt->type() );
   fassert( m_st != 0 );

   bool result = checkSymbols();

   // if the pareser is not interactive, append the statement even after undefined errors.
   if( result || ! m_parser->interactive() )
   {
      m_st->append(stmt);
      onNewStatement( stmt );
   }
   else
   {
      // when interactive, we don't want to have useless statements.
      delete stmt;
   }
}


void ParserContext::openBlock( Statement* parent, SynTree* branch )
{
   TRACE("ParserContext::openBlock type %d", (int) parent->type() );

   bool result = checkSymbols();

   // if the pareser is not interactive, append the statement even after undefined errors.
   if( result || ! m_parser->interactive() )
   {
      m_cstatement = parent;
      m_st = branch;
      _p->m_frames.push_back( CCFrame(parent, branch) );
   }
   else
   {
      // when interactive, we don't want to have useless statements.
      delete parent;
   }
}


SynTree* ParserContext::changeBranch()
{
   TRACE("ParserContext::changeBranch", 0 );

   bool result = checkSymbols();

   // if the pareser is not interactive, append the statement even after undefined errors.
   if( result || ! m_parser->interactive() )
   {
      m_st = new SynTree;
      _p->m_frames.back().m_branch = m_st;
      return m_st;
   }
   else
   {
      // when interactive, we don't want to have useless statements.
      return 0;
   }
}


void ParserContext::openFunc( SynFunc *func, GlobalSymbol *gs )
{
   TRACE("ParserContext::openFunc -- %s", func->name().c_ize() );
   m_cfunc = func;
   m_st = &func->syntree();
   m_cstatement = 0;
   _p->m_frames.push_back(CCFrame(func, gs));

   // get the symbol table.
   m_symtab = &func->symbols();
}


void ParserContext::openClass( Class *cls, bool bIsObject, GlobalSymbol *gs )
{
   TRACE("ParserContext::openClass -- depth %d %s%s", _p->m_frames.size() + 1,
            cls->name().c_ize(), bIsObject ? "":" (object)" );
   m_cclass = cls;
   m_cstatement = 0;
   _p->m_frames.push_back(CCFrame(cls, bIsObject, gs));
   // TODO: get the symbol table.
}


void ParserContext::closeContext()
{
   TRACE("ParserContext::closeContext -- closing context on depth %d", _p->m_frames.size() );
   fassert( !_p->m_frames.empty() );
   
   // copy by value
   CCFrame bframe = _p->m_frames.back();

   // as we're removing the frame.
   _p->m_frames.pop_back();
   // we can never close the main context
   fassert( ! _p->m_frames.empty() );
   if( bframe.m_bStatePushed )
   {
      m_parser->popState();
   }

   m_st = _p->m_frames.back().m_branch;
   // updating the syntactic tree
   Private::FrameVector::const_reverse_iterator riter = _p->m_frames.rbegin();
   m_symtab = 0; // if we don't find a parent context, the symtab is 0
   while( riter != _p->m_frames.rbegin() )
   {
      const CCFrame& curFrame = *riter;
      switch( curFrame.m_type )
      {
         case CCFrame::t_object_type:
         case CCFrame::t_class_type:
            // todo
            //m_symtab = curFrame.m_elem.cls;
            riter = _p->m_frames.rbegin(); // we're done, froce break;
            break;

         case CCFrame::t_func_type:
            m_symtab = &curFrame.m_elem.func->symbols();
            riter = _p->m_frames.rbegin(); // we're done, froce break;
            break;

         default:
            // keep the current symtab.
            m_symtab = m_symtab; // to make compilers happy
            // continue the loop
      }
   }
   
   // notify the new item.
   switch( bframe.m_type )
   {
      case CCFrame::t_none_type: fassert(0); break;

      // if it's a base, there's nothing to do (but it's also strange...)
      case CCFrame::t_base_type: break;

      // if it's a class...
      case CCFrame::t_object_type:
      case CCFrame::t_class_type: 
         {
            Private::FrameVector::const_reverse_iterator riter = _p->m_frames.rbegin();
            // find the new topmost class
            m_cclass = 0;
            while( riter != _p->m_frames.rend() )
            {
               if ( riter->m_type == CCFrame::t_class_type || riter->m_type == CCFrame::t_object_type )
               {
                  m_cclass = riter->m_elem.cls;
                  break;
               }

               ++riter;
            }
         }
         onNewClass( bframe.m_elem.cls, bframe.m_type == CCFrame::t_object_type, bframe.m_sym );
         break;

      case CCFrame::t_func_type:
         {
            Private::FrameVector::const_reverse_iterator riter = _p->m_frames.rbegin();
            // find the new topmost class
            m_cfunc = 0;
            while( riter != _p->m_frames.rend() )
            {
               if( riter->m_type == CCFrame::t_class_type || riter->m_type == CCFrame::t_object_type )
               {
                  // classes before functions
                  break;
               }

               if ( riter->m_type == CCFrame::t_func_type  )
               {
                  m_cfunc = riter->m_elem.func;
                  break;
               }

               ++riter;
            }
         }
         onNewFunc( bframe.m_elem.func, bframe.m_sym );
         break;

      case CCFrame::t_stmt_type:
         {
            Private::FrameVector::const_reverse_iterator riter = _p->m_frames.rbegin();
            // find the new topmost class
            m_cstatement = 0;
            while( riter != _p->m_frames.rend() )
            {
               if( riter->m_type == CCFrame::t_class_type || 
                   riter->m_type == CCFrame::t_object_type ||
                   riter->m_type == CCFrame::t_func_type )
               {
                  // classes or functions before block statements.
                  break;
               }

               if ( riter->m_type != CCFrame::t_func_type  )
               {
                  m_cstatement = riter->m_elem.stmt;
                  break;
               }

               ++riter;
            }
         }
         addStatement( bframe.m_elem.stmt ); // will also do onNewStatement
         break;
   }
}

void ParserContext::reset()
{
   _p->m_frames.clear();
   _p->m_unknown.clear();

   m_st = 0;
   m_cstatement = 0;
   m_cfunc = 0;
   m_cclass = 0;
   m_symtab = 0;
}

}

/* end of compcontext.cpp */

