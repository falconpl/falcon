/*
   FALCON - The Falcon Programming Language.
   FILE: parsercontext.cpp

   Compilation context for Falcon source file compilation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 16 Apr 2011 18:17:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parsercontext.cpp"

#include <falcon/trace.h>
#include <falcon/symbol.h>
#include <falcon/synfunc.h>
#include <falcon/falconclass.h>
#include <falcon/error.h>
#include <falcon/class.h>
#include <falcon/falconclass.h>

#include <falcon/sp/parsercontext.h>
#include <falcon/sp/sourceparser.h>

#include <falcon/psteps/exprsym.h>

#include <vector>
#include <map>
#include <list>
#include <deque>


namespace Falcon {

//==================================================================
// Frame class
//

class ParserContext::CCFrame
{
   typedef union tag_elem {
      FalconClass* cls;
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
   CCFrame( FalconClass* cls, bool bIsObject, Symbol* gs );
   CCFrame( SynFunc* func, Symbol* gs );
   CCFrame( Statement* stmt, bool bAutoClose = false );

   void setup();

public:
   friend class ParserContext;

   /** Syntree element topping this frame. */
   t_elem m_elem;

   /** Type of frame */
   t_type m_type;

   /** True if a parser state was pushed at this frame */
   bool m_bStatePushed;
   
   /** True when the context should be closed immediately at first addStatement */
   bool m_bAutoClose;

   Symbol* m_sym;

   //===================================================

   // Pre-cached syntree for performance.
   SynTree* m_st;

   // Current function, precached for performance.
   Statement* m_cstatement;

   // Current function, precached for performance.
   SynFunc * m_cfunc;

   // Current class, precached for performance.
   FalconClass * m_cclass;

   // Current symbol table, precached for performance.
   SymbolTable* m_symtab;
};

ParserContext::CCFrame::CCFrame():
   m_type( t_none_type ),
   m_bStatePushed( false ),
   m_bAutoClose(false),
   m_sym(0)
{
   setup();
}

ParserContext::CCFrame::CCFrame( FalconClass* cls, bool bIsObject, Symbol* gs ):
   m_type( bIsObject ? t_object_type : t_class_type ),
   m_bStatePushed( false ),
   m_bAutoClose( false ),
   m_sym(gs)
{
   m_elem.cls = cls;
   setup();
}

ParserContext::CCFrame::CCFrame( SynFunc* func, Symbol* gs ):
   m_type( t_func_type ),
   m_bStatePushed( false ),
   m_bAutoClose( false ),
   m_sym(gs)
{
   m_elem.func = func;
   setup();
}


ParserContext::CCFrame::CCFrame( Statement* stmt, bool bAutoClose ):
   m_type( t_stmt_type ),
   m_bStatePushed( false ),
   m_bAutoClose( bAutoClose ),
   m_sym(0)
{
   m_elem.stmt = stmt;
   setup();
}

void ParserContext::CCFrame::setup()
{
   m_st = 0;
   m_cstatement = 0;
   m_cfunc = 0;
   m_cclass = 0;
   m_symtab = 0;
}
//==================================================================
// Main parser context class
//

class ParserContext::Private
{
private:
   friend class ParserContext;

   typedef std::list< ExprSymbol* > UnknownList;
   typedef std::list< UnknownList > UnknownsStack;

   /** Map of unknown symbols created during the current statement parsing. */
   UnknownsStack m_unknown;

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


void ParserContext::saveStatus( CCFrame& cf ) const
{
   cf.m_st = m_st;
   cf.m_cstatement = m_cstatement;
   cf.m_cfunc = m_cfunc;
   cf.m_cclass = m_cclass;
   cf.m_symtab = m_symtab;
}


void ParserContext::restoreStatus( const CCFrame& cf )
{
   m_st = cf.m_st;
   m_cstatement = cf.m_cstatement;
   m_cfunc = cf.m_cfunc;
   m_cclass = cf.m_cclass;
   m_symtab = cf.m_symtab;
}


void ParserContext::openMain( SynTree* st )
{
   _p->m_frames.push_back( CCFrame() );
   m_st = st;
   saveStatus( _p->m_frames.back() );
}

void ParserContext::onStatePushed( bool isPushedState )
{
   fassert( ! _p->m_frames.empty() );
   if( isPushedState )
   {
      _p->m_frames.back().m_bStatePushed = true;
   }
   _p->m_unknown.push_back(Private::UnknownList());
}

void ParserContext::onStatePopped()
{
   _p->m_unknown.pop_back();
}



ExprSymbol* ParserContext::addVariable( const String& variable )
{
   TRACE("ParserContext::addVariable %s", variable.c_ize() );

   ExprSymbol* expr = new ExprSymbol( variable );   
   _p->m_unknown.back().push_back( expr );
   return expr;
}

Symbol* ParserContext::addDefineSymbol( const String& variable )
{
   Symbol* nuks;
   
   if( m_symtab == 0 )
   {
      // we're in the global context.
      bool bAlready;
      nuks = onGlobalDefined( variable, bAlready );
   }
   else
   {
      // add it in the current symbol table.
      nuks = m_symtab->addLocal( variable );
   }
   
   return nuks;
}

void ParserContext::undoVariable( const String& variable )
{
   TRACE("ParserContext::undoVariable %s", variable.c_ize() );
   fassert( ! _p->m_unknown.empty() );
   
   if( _p->m_unknown.empty() )
   {
      //todo error!
      return;
   }

   Private::UnknownList& ul = _p->m_unknown.back();
   Private::UnknownList::iterator iter = ul.begin();
   
   while( iter != ul.end() )
   {
      if ((*iter)->name() == variable )
      {
         ul.erase(iter);
         return;
      }
      ++iter;
   }
}

void ParserContext::abandonSymbols()
{
   MESSAGE( "ParserContext::abandonSymbols" );

   fassert( ! _p->m_unknown.empty() );
   
   if( _p->m_unknown.empty() )
   {
      //todo error!
      return;
   }

   _p->m_unknown.back().clear();
}

void ParserContext::defineSymbols( Expression* expr )
{
   TRACE("ParserContext::defineSymbols on (: %s :)", expr->describe().c_ize() );

   if( expr->trait() == Expression::e_trait_symbol )
   {
      ExprSymbol* exprsym = static_cast<ExprSymbol*>(expr);
      
      Symbol* uks = exprsym->symbol();
      if( uks == 0 )
      {
         TRACE1("ParserContext::defineSymbol trying to define symbol \"%s\"", exprsym->name().c_ize() );
         Symbol* nuks = findSymbol(exprsym->name());
         // Found?
         if( nuks == 0 )
         {
            // --no ? create it
            if( m_symtab == 0 )
            {
               // we're in the global context.
               bool bAlready;
               nuks = onGlobalDefined( exprsym->name(), bAlready );            
            }
            else
            {
               // add it in the current symbol table.
               nuks = m_symtab->addLocal( exprsym->name() );
            }
         }

         // created?
         if( nuks != 0 )
         {
            TRACE("ParserContext::defineSymbol defined \"%s\" as %d",
                  nuks->name().c_ize(), nuks->type() );
            // remove from the unknowns
            undoVariable( exprsym->name() );
            exprsym->safeGuard( nuks );
         }
         else
         {
            //TODO: Use a more fitting error code for this?
            m_parser->addError(e_undef_sym, m_parser->currentSource(), exprsym->line(), exprsym->chr(), 0, uks->name() );

            TRACE("ParserContext::defineSymbol cannot define \"%s\"",
                  exprsym->name().c_ize() );
         }
      }
      else
      {
         // else, the symbol is already ok.
         TRACE2("ParserContext::defineSymbol \"%s\" already of type %d",
                  uks->name().c_ize(), uks->type() );
      }
   }
}


bool ParserContext::checkSymbols()
{
   bool isOk = true;
   Private::UnknownList& unknown = _p->m_unknown.back();

   // try with all the undefined symbols.
   TRACE("ParserContext::checkSymbols on %d syms", (int) unknown.size() );
   Private::UnknownList::iterator iter = unknown.begin();
   typedef std::deque< std::pair<String, int> > UnkonwnList;
   UnkonwnList unknownNames;

   while( iter != unknown.end() )
   {
      ExprSymbol* esym = *iter;
      if ( esym->symbol() != 0 )
      {
         // already defined.
         ++iter;
         continue;
      }
      
      Symbol* new_sym = findSymbol( esym->name() );

      if ( new_sym == 0 )
      {
         // see if it's a global or extern for our sub-class...
         TRACE1("ParserContext::checkSymbols \"%s\" is undefined, up-notifying", esym->name().c_ize() );
         new_sym = onUndefinedSymbol( esym->name() );

         // still undefined? -- we surrender, and hope that the subclass has properly raised
         if ( new_sym == 0 )
         {
            TRACE1("ParserContext::checkSymbols \"%s\" leaving this symbol undefined", esym->name().c_ize() );
         }
      }

      if( new_sym != 0 )
      {
         TRACE1("ParserContext::checkSymbols \"%s\" is now of type %d", esym->name().c_ize(), new_sym->type() );
         esym->safeGuard(new_sym);
      }
      else
      {
         TRACE1("ParserContext::checkSymbols cannot define \"%s\"",
                  esym->name().c_ize() );

         String name = esym->name();
         int dat = esym->line();
         // this will probably destroy the symbol.
         // record for later error generation
         unknownNames.push_back( std::make_pair( name, dat ) );
         // we know that the symbol is lost
         // add error will clear unknown symbols. return immediately after this call, iterators are no longer valid.
         isOk = false;

         // -- see if the callee wants to do something about that
      }
      ++iter;
   }

   // we don't want addError to clear the unknown symbols stack.
   unknown.clear();

   UnkonwnList::iterator un_iter = unknownNames.begin();
   while( un_iter != unknownNames.end() )
   {
      m_parser->addError(e_undef_sym, m_parser->currentSource(),
         un_iter->second , 0, 0, un_iter->first );
      ++un_iter;
   }

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
         if( sym->type() == Symbol::e_st_local )
         {
            TRACE1("ParserContext::findSymbol \"%s\" found, need to be closed", sym->name().c_ize() );
            //TODO: Properly close symbols. -- this won't work
            Symbol* closym = m_symtab->addClosed(name);
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
   TRACE("ParserContext::addStatement type '%s'",
         stmt->handler() == 0 ? "none": 
          stmt->handler()->name().c_ize() );
   fassert( m_st != 0 );

   bool result = checkSymbols();

   // if the pareser is not interactive, append the statement even after undefined errors.
   if( result || ! m_parser->interactive() )
   {
      m_st->append(stmt);
      onNewStatement( stmt );
      
      if( _p->m_frames.back().m_bAutoClose )
      {
         closeContext();
      }
   }
   else
   {
      // when interactive, we don't want to have useless statements.
      delete stmt;
   }
}


void ParserContext::openBlock( Statement* parent, SynTree* branch, bool bAutoClose )
{
   TRACE("ParserContext::openBlock type '%s'",
         parent->handler() == 0 ? "none": 
          parent->handler()->name().c_ize() );

   saveStatus( _p->m_frames.back() );

   bool result = parent->discardable() ? true : checkSymbols();

   // if the pareser is not interactive, append the statement even after undefined errors.
   if( result || ! m_parser->interactive() )
   {
      _p->m_frames.push_back( CCFrame(parent, bAutoClose) );
      m_cstatement = parent;
      m_st = branch;
   }
   else
   {
      // when interactive, we don't want to have useless statements.
      delete parent;
   }
}


SynTree* ParserContext::changeBranch()
{
   MESSAGE( "ParserContext::changeBranch" );

   bool result = checkSymbols();

   // if the pareser is not interactive, append the statement even after undefined errors.
   if( result || ! m_parser->interactive() )
   {
      m_st = new SynTree;
      return m_st;
   }
   else
   {
      // when interactive, we don't want to have useless statements.
      return 0;
   }
}


void ParserContext::openFunc( SynFunc *func, Symbol *gs )
{
   TRACE("ParserContext::openFunc -- %s", func->name().c_ize() );

   saveStatus( _p->m_frames.back() );

   m_cfunc = func;
   m_st = &func->syntree();
   m_cstatement = 0;
   _p->m_frames.push_back(CCFrame(func, gs));

   // get the symbol table.
   m_symtab = &func->symbols();
}


void ParserContext::openClass( Class* cls, bool bIsObject, Symbol *gs )
{
   TRACE("ParserContext::openClass -- depth %d %s%s", (int)_p->m_frames.size() + 1,
            cls->name().c_ize(), bIsObject ? "":" (object)" );

   saveStatus( _p->m_frames.back() );

   // we know we're compiling source classes.
   FalconClass* fcls = static_cast<FalconClass*>(cls);

   m_cclass = fcls;
   m_cstatement = 0;
   _p->m_frames.push_back(CCFrame(fcls, bIsObject, gs));   
   m_symtab = &fcls->makeConstructor()->symbols();
   // TODO: get the symbol table.
}


void ParserContext::closeContext()
{
   TRACE("ParserContext::closeContext -- closing context on depth %d", (int)_p->m_frames.size() );
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

   // restore our previous status
   restoreStatus(_p->m_frames.back());

   // updating the syntactic tree
   // Private::FrameVector::const_reverse_iterator riter = _p->m_frames.rbegin();
  
   // notify the new item.
   switch( bframe.m_type )
   {
      case CCFrame::t_none_type: fassert(0); break;

      // if it's a base, there's nothing to do (but it's also strange...)
      case CCFrame::t_base_type: break;

      // if it's a class...
      case CCFrame::t_object_type:
      case CCFrame::t_class_type:
         // TODO: allow nested classes.
         onNewClass( bframe.m_elem.cls, bframe.m_type == CCFrame::t_object_type, bframe.m_sym );
         break;

      case CCFrame::t_func_type:
         // is this a method?
         if ( m_cclass != 0 )
         {
            m_cclass->addMethod( bframe.m_elem.func );
         }
         else
         {
            onNewFunc( bframe.m_elem.func, bframe.m_sym );
         }
         break;

      case CCFrame::t_stmt_type:
         if( ! bframe.m_elem.stmt->discardable() )
         {
            addStatement( bframe.m_elem.stmt ); // will also do onNewStatement
         }
         else
         {
            delete bframe.m_elem.stmt;
         }
         break;
   }
}


bool ParserContext::isTopLevel() const
{
   return _p->m_frames.size() < 2;
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

/* end of parsercontext.cpp */
