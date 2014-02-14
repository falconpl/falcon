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
#include <falcon/psteps/exprlit.h>

#include <vector>
#include <map>
#include <list>
#include <deque>

#include "falcon/psteps/exprcompare.h"


namespace Falcon {

//==================================================================
// Frame class
//

class ParserContext::CCFrame
{
   typedef union tag_elem {
      FalconClass* cls;
      SynFunc* func;
      TreeStep* stmt;
      SynTree* synTree;
      void* raw;
   } t_elem;

   typedef enum tag_type {
      t_none_type,
      t_class_type,
      t_object_type,
      t_func_type,
      t_static_func_type,
      t_stmt_type,
      t_temp_type,
      t_base_type
   } t_type;

   CCFrame();
   CCFrame( FalconClass* cls, bool bIsObject );
   CCFrame( SynFunc* func, bool isStatic = false );
   CCFrame( TreeStep* stmt, bool bAutoClose = false, bool bAutoAdd = true );
   CCFrame( SynTree* st );

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

   /** Automatically add the statement when closed */
   bool m_bAutoAdd;

   //===================================================

   // Pre-cached syntree for performance.
   SynTree* m_st;

   // Current function, precached for performance.
   TreeStep* m_cstatement;

   // Current function, precached for performance.
   SynFunc * m_cfunc;

   // Current class, precached for performance.
   FalconClass * m_cclass;
};

ParserContext::CCFrame::CCFrame():
   m_type( t_none_type ),
   m_bStatePushed( false ),
   m_bAutoClose(false),
   m_bAutoAdd( true )
{
   setup();
}

ParserContext::CCFrame::CCFrame( FalconClass* cls, bool bIsObject ):
   m_type( bIsObject ? t_object_type : t_class_type ),
   m_bStatePushed( false ),
   m_bAutoClose( false ),
   m_bAutoAdd( true )
{
   m_elem.cls = cls;
   setup();
}

ParserContext::CCFrame::CCFrame( SynFunc* func, bool bIsStatic ):
   m_type( bIsStatic ? t_static_func_type : t_func_type ),
   m_bStatePushed( false ),
   m_bAutoClose( false ),
   m_bAutoAdd( true )
{
   m_elem.func = func;
   setup();
}


ParserContext::CCFrame::CCFrame( TreeStep* stmt, bool bAutoClose, bool bAutoAdd ):
   m_type( t_stmt_type ),
   m_bStatePushed( false ),
   m_bAutoClose( bAutoClose ),
   m_bAutoAdd( bAutoAdd )
{
   m_elem.stmt = stmt;
   setup();
}

ParserContext::CCFrame::CCFrame( SynTree* oldST ):
   m_type( t_temp_type ),
   m_bStatePushed( false ),
   m_bAutoClose( true ),
   m_bAutoAdd( true )
{
   m_elem.synTree = oldST;
   setup();
}


void ParserContext::CCFrame::setup()
{
   m_st = 0;
   m_cstatement = 0;
   m_cfunc = 0;
   m_cclass = 0;
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

   typedef std::deque<ExprLit*> LitContexts;
   LitContexts m_litContexts;
   
   Private() {}
   ~Private() {
      // do not delete lit contexts; they are in the parser token stack.
   }
};


ParserContext::ParserContext( SourceParser* sp ):
   m_parser(sp),
   m_st(0),
   m_cstatement(0),
   m_cfunc(0),
   m_cclass(0)
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
}


void ParserContext::restoreStatus( const CCFrame& cf )
{
   m_st = cf.m_st;
   m_cstatement = cf.m_cstatement;
   m_cfunc = cf.m_cfunc;
   m_cclass = cf.m_cclass;
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


void ParserContext::defineSymbol( const String& variable )
{
   TRACE("ParserContext::defineSymbol(: %s :)", variable.c_ize() );
   
   // in literal context, we never define variables.
   if( !_p->m_litContexts.empty() ) {
      return;
   }

   if( m_cfunc == 0 )
   {
      // we're in the global context.
      bool bAlready;
      onGlobalDefined( variable, bAlready );
   }
   else
   {
      // add it in the current symbol table.
      m_cfunc->locals().insert( variable );
      // ignore symbol duplication.
   }
}

bool ParserContext::accessSymbol( const String& variable )
{
   TRACE("ParserContext::accessSymbol(: %s :)", variable.c_ize() );

   if( m_cfunc == 0 && m_cclass == 0 )
   {
      // we're in the global context.
      if( _p->m_litContexts.empty() )
      {
         return onGlobalAccessed( variable );
      }
      else {
         // don't need to add any local.
         TRACE1("ParserContext::accessSymbol(: %s :) ignoring access in literal contexts", variable.c_ize() );
      }
   }
   else if( m_cclass != 0 )
   {
      // it's a property.
      if( m_cclass->constructor() != 0
          && m_cclass->constructor()->parameters().find(variable) >= 0
        )
      {
         TRACE1("ParserContext::accessSymbol(: %s :) access in class property is a class parameter", variable.c_ize() );
      }
      else {
         bool isLocal = isLocalSymbol( variable );
         if( isLocal )
         {
            TRACE1("ParserContext::accessSymbol(: %s :) unknown property found in a local context", variable.c_ize() );
         }
         else {
            TRACE1("ParserContext::accessSymbol(: %s :) unknown property, importing.", variable.c_ize() );
            return onGlobalAccessed(variable);
         }
      }
   }
   else
   {
      // search in the local contexts
      bool isLocal = isLocalSymbol( variable );
      // not found?

      if ( ! isLocal )
      {
         if( !_p->m_litContexts.empty() )
         {
            m_cfunc->locals().insert(variable);
            TRACE1("ParserContext::accessSymbol(: %s :) found a local symbol", variable.c_ize() );
         }
         else if( isParentLocal( variable ) ) {
            m_cfunc->closed().insert(variable);
            TRACE1("ParserContext::accessSymbol(: %s :) closed symbol", variable.c_ize() );
         }
         else {
            // tell the subclass we're accessing this variable as global.
            return onGlobalAccessed( variable );
         }
      }
   }

   // only access to globals can return "non local"
   // -- and we returned directly any onGlobalAccessed.
   return true;
}


void ParserContext::defineSymbols( Expression* expr )
{
   TRACE("ParserContext::defineSymbols on (: %s :)", expr->describe().c_ize() );

   if( expr->trait() == Expression::e_trait_symbol )
   {
      ExprSymbol* exprsym = static_cast<ExprSymbol*>( expr );
      const Symbol* sym = exprsym->symbol();
      defineSymbol( sym->name() );
   }
   else {
      uint32 arity = expr->arity();
      for( uint32 i = 0; i < arity; ++ i ) {
         // expressions can only have expressions as nth()
         Expression* child = static_cast<Expression*>(expr->nth(i));
         if( child != 0 && child->trait() != Expression::e_trait_composite )
         {
            if( expr->fullDefining() )
            {
               defineSymbols( child );
            }
            else {
               accessSymbols( child );
            }
         }
      }
   }
}


void ParserContext::accessSymbols( Expression* expr )
{
   TRACE("ParserContext::accessSymbols on (: %s :)", expr->describe().c_ize() );

   if( expr->trait() == Expression::e_trait_composite ) {
      return;
   }

   if( expr->trait() == Expression::e_trait_symbol )
   {
      ExprSymbol* exprsym = static_cast<ExprSymbol*>( expr );
      if( !exprsym->isPure() )
      {
         const Symbol* sym = exprsym->symbol();
         accessSymbol( sym->name() );
      }
   }
   else {
      uint32 arity = expr->arity();
      for( uint32 i = 0; i < arity; ++ i ) {
         // expressions can only have expressions as nth()
         Expression* child = static_cast<Expression*>(expr->nth(i));
         if( child != 0 && child->category() == TreeStep::e_cat_expression )
         {
            accessSymbols( child );
         }
      }
   }
}


bool ParserContext::isLocalSymbol( const String& name )
{
   TRACE1("ParserContext::findLocalSymbol \"%s\"", name.c_ize() );
   if( m_cfunc == 0 )
   {
      return false;
   }

   // found at first shot?
   int32 symId =  m_cfunc->parameters().find( name );
   if( symId < 0 )
   {
      symId = m_cfunc->locals().find( name );
   }

   if( symId >=  0 )
   {
      TRACE1("ParserContext::findSymbol \"%s\" found locally", name.c_ize() );
      return true;
   }

   return false;
}

bool ParserContext::isParentLocal( const String& name )
{
   Private::FrameVector::const_reverse_iterator iter = _p->m_frames.rbegin();
   while( iter != _p->m_frames.rend() )
   {
      const CCFrame& frame = *iter;

      // skip symbol table of classes, they can't be referenced by inner stuff.
      if( frame.m_type == CCFrame::t_class_type )
      {
         // we can't close symbols across class definitions.
         break;
      }

      if( frame.m_type != CCFrame::t_func_type || frame.m_elem.func == m_cfunc ) {
         ++iter;
         continue;
      }

      int32 symId =  frame.m_elem.func->parameters().find( name );
      if( symId < 0 )
      {
         symId = frame.m_elem.func->locals().find( name );
      }

      if( symId >=  0 )
      {
         TRACE1("ParserContext::findSymbol \"%s\" found, need to be closed", name.c_ize() );
         return true;
      }
      ++iter;
   }

   return false;
}


void ParserContext::addStatement( TreeStep* stmt )
{
   TRACE("ParserContext::addStatement type '%s'",
         stmt->handler() == 0 ? "none": 
          stmt->handler()->name().c_ize() );
   fassert( m_st != 0 );

   // if the pareser is not interactive, append the statement even after undefined errors.
   if( ! m_parser->hasErrors() || ! m_parser->interactive() )
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

void ParserContext::openLitContext( ExprLit* el ) {
   _p->m_litContexts.push_back(el);
   //m_varmap = el->varmap();
}

ExprLit* ParserContext::closeLitContext() 
{
   if( ! _p->m_litContexts.empty() )
   {
      ExprLit* current = _p->m_litContexts.back();
      _p->m_litContexts.pop_back();

      return current;
   }
   
   return 0;
}

ExprLit* ParserContext::currentLitContext() 
{
   if( ! _p->m_litContexts.empty() ) {
      return _p->m_litContexts.back();
   }
   
   return 0;
}


bool ParserContext::isLitContext() const
{
   return ! _p->m_litContexts.empty();
}

bool ParserContext::isGlobalContext() const
{
   return _p->m_litContexts.empty() && currentClass() == 0 && currentFunc() == 0;
}

void ParserContext::openBlock( TreeStep* parent, SynTree* branch, bool bAutoClose, bool bAutoAdd )
{
   TRACE("ParserContext::openBlock type '%s'",
         parent->handler() == 0 ? "none": 
          parent->handler()->name().c_ize() );

   saveStatus( _p->m_frames.back() );

   //bool result = parent->discardable() ? true : checkSymbols();

   _p->m_frames.push_back( CCFrame(parent, bAutoClose, bAutoAdd ) );
   m_cstatement = parent;
   m_st = branch;
}

void ParserContext::openTempBlock( SynTree* oldBranch, SynTree* newBranch )
{   
   MESSAGE("ParserContext::openTempBlock");

   saveStatus( _p->m_frames.back() );

   // if the pareser is not interactive, append the statement even after undefined errors.
   _p->m_frames.push_back( CCFrame( oldBranch ) );
   m_st = newBranch;
}


SynTree* ParserContext::changeBranch()
{
   MESSAGE( "ParserContext::changeBranch" );

   // if the pareser is not interactive, append the statement even after undefined errors.
   if( ! m_parser->hasErrors() || ! m_parser->interactive() )
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


void ParserContext::changeBranch( SynTree* st)
{
   MESSAGE( "ParserContext::changeBranch 2" );
   m_st = st;
}


void ParserContext::openFunc( SynFunc *func, bool bIsStatic )
{
   TRACE("ParserContext::openFunc -- %s", func->name().c_ize() );

   saveStatus( _p->m_frames.back() );

   m_cfunc = func;
   m_st = &func->syntree();
   m_cstatement = 0;
   _p->m_frames.push_back(CCFrame(func, bIsStatic));
}


void ParserContext::openClass( Class* cls, bool bIsObject )
{
   TRACE("ParserContext::openClass -- depth %d %s%s", (int)_p->m_frames.size() + 1,
            cls->name().c_ize(), bIsObject ? "":" (object)" );

   saveStatus( _p->m_frames.back() );

   // we know we're compiling source classes.
   FalconClass* fcls = static_cast<FalconClass*>(cls);

   m_cclass = fcls;
   m_cstatement = 0;
   _p->m_frames.push_back(CCFrame(fcls, bIsObject ));
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
      
      // If it's a temporary frame, restore the requested tree.
      case CCFrame::t_temp_type:
         if( bframe.m_elem.synTree != 0 ) {
            m_st = bframe.m_elem.synTree;
         }
         break;

      // if it's a class...
      case CCFrame::t_object_type:
      case CCFrame::t_class_type:
         // TODO: allow nested classes.
         onCloseClass( bframe.m_elem.cls, bframe.m_type == CCFrame::t_object_type );
         break;

      case CCFrame::t_func_type:
      case CCFrame::t_static_func_type:
         // is this a method?
         if ( m_cclass != 0 && currentFunc() == 0 )
         {
            // unless it's the constructor -- in which case it's already added
            if ( bframe.m_elem.func->methodOf() != m_cclass )
            {
               m_cclass->addMethod( bframe.m_elem.func, bframe.m_type ==  CCFrame::t_static_func_type );
            }
         }

         onCloseFunc( bframe.m_elem.func );
         break;

      case CCFrame::t_stmt_type:
      {
         TreeStep* step = bframe.m_elem.stmt;
         if( step->category() == TreeStep::e_cat_statement )
         {
            Statement* stmt = static_cast<Statement*>(step);

            if( ! stmt->discardable() )
            {
               stmt->minimize();
               if ( bframe.m_bAutoAdd )
               {
                  addStatement( stmt ); // will also do onNewStatement
               }
            }
            else
            {
               delete bframe.m_elem.stmt;
            }
         }
         else {
            if ( bframe.m_bAutoAdd )
            {
               addStatement( step ); // will also do onNewStatement
            }
         }
      }
         break;
   }
}



void ParserContext::dropContext()
{
   TRACE("ParserContext::dropContext -- closing context on depth %d", (int)_p->m_frames.size() );
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
}


Item* ParserContext::getValue( const String& name )
{
   const Symbol* sym = Engine::getSymbol(name);
   try
   {
      Item* item = getValue( sym );
      sym->decref();
      return item;
   }
   catch( ... )
   {
      sym->decref();
      throw;
   }

   return 0;
}

}

/* end of parsercontext.cpp */
