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
      Statement* stmt;
      SynTree* synTree;
      void* raw;
   } t_elem;

   typedef enum tag_type {
      t_none_type,
      t_class_type,
      t_object_type,
      t_func_type,
      t_stmt_type,
      t_temp_type,
      t_base_type
   } t_type;

   CCFrame();
   CCFrame( FalconClass* cls, bool bIsObject );
   CCFrame( SynFunc* func );
   CCFrame( Statement* stmt, bool bAutoClose = false );
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
   VarMap* m_varmap;
};

ParserContext::CCFrame::CCFrame():
   m_type( t_none_type ),
   m_bStatePushed( false ),
   m_bAutoClose(false)
{
   setup();
}

ParserContext::CCFrame::CCFrame( FalconClass* cls, bool bIsObject ):
   m_type( bIsObject ? t_object_type : t_class_type ),
   m_bStatePushed( false ),
   m_bAutoClose( false )
{
   m_elem.cls = cls;
   setup();
}

ParserContext::CCFrame::CCFrame( SynFunc* func ):
   m_type( t_func_type ),
   m_bStatePushed( false ),
   m_bAutoClose( false )
{
   m_elem.func = func;
   setup();
}


ParserContext::CCFrame::CCFrame( Statement* stmt, bool bAutoClose ):
   m_type( t_stmt_type ),
   m_bStatePushed( false ),
   m_bAutoClose( bAutoClose )
{
   m_elem.stmt = stmt;
   setup();
}

ParserContext::CCFrame::CCFrame( SynTree* oldST ):
   m_type( t_temp_type ),
   m_bStatePushed( false ),
   m_bAutoClose( true )
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
   m_varmap = 0;
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
   m_cclass(0),
   m_varmap(0)
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
   cf.m_varmap = m_varmap;
}


void ParserContext::restoreStatus( const CCFrame& cf )
{
   m_st = cf.m_st;
   m_cstatement = cf.m_cstatement;
   m_cfunc = cf.m_cfunc;
   m_cclass = cf.m_cclass;
   m_varmap = cf.m_varmap;
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


Variable* ParserContext::defineSymbol( const String& variable )
{
   TRACE("ParserContext::defineSymbol on (: %s :)", variable.c_ize() );
   Variable* nuks;
   
   if( m_varmap == 0 )
   {
      // we're in the global context.
      bool bAlready;
      nuks = onGlobalDefined( variable, bAlready );
   }
   else
   {
      // add it in the current symbol table.
      nuks = m_varmap->addLocal( variable );
   }
   
   return nuks;
}

Variable* ParserContext::accessSymbol( const String& variable )
{
   TRACE("ParserContext::accessSymbol on (: %s :)", variable.c_ize() );
   Variable* nuks;
   
   if( m_varmap == 0 )
   {
      // we're in the global context.
      nuks = onGlobalAccessed( variable );
   }
   else
   {
      // search in the local contexts
      nuks = findSymbol( variable );
      // not found?

      if ( nuks == 0 ) {
         // tell the subclass we're accessing this variable as global.
         nuks = onGlobalAccessed( variable );
      }
   }

   return nuks;
}


void ParserContext::defineSymbols( Expression* expr )
{
   TRACE("ParserContext::defineSymbols on (: %s :)", expr->describe().c_ize() );

   if( expr->trait() == Expression::e_trait_symbol )
   {
      ExprSymbol* exprsym = static_cast<ExprSymbol*>( expr );
      Symbol* sym = exprsym->symbol();
      defineSymbol( sym->name() );
   }
   else {
      uint32 arity = expr->arity();
      for( uint32 i = 0; i < arity; ++ i ) {
         // expressions can only have expressions as nth()
         Expression* expr = static_cast<Expression*>(expr->nth(i));
         if( expr->trait() != Expression::e_trait_composite )
         {
            defineSymbols( expr );
         }
      }
   }
}


void ParserContext::accessSymbols( Expression* expr )
{
   TRACE("ParserContext::accessSymbols on (: %s :)", expr->describe().c_ize() );

   if( expr->trait() == Expression::e_trait_symbol )
   {
      ExprSymbol* exprsym = static_cast<ExprSymbol*>( expr );
      Symbol* sym = exprsym->symbol();
      accessSymbol( sym->name() );
   }
   else {
      uint32 arity = expr->arity();
      for( uint32 i = 0; i < arity; ++ i ) {
         // expressions can only have expressions as nth()
         Expression* child = static_cast<Expression*>(expr->nth(i));
         if( child->trait() != Expression::e_trait_composite )
         {
            accessSymbols( child );
         }
      }
   }
}


Variable* ParserContext::findSymbol( const String& name )
{
   TRACE1("ParserContext::findSymbol \"%s\"", name.c_ize() );
   if( m_varmap == 0 )
   {
      return 0;
   }

   // found at first shot?
   Variable* var = m_varmap->find( name );
   if( var !=  0 )
   {
      TRACE1("ParserContext::findSymbol \"%s\" found locally", name.c_ize() );
      return var;
   }

   Private::FrameVector::const_reverse_iterator iter = _p->m_frames.rbegin();
   while( iter != _p->m_frames.rend() )
   {
      const CCFrame& frame = *iter;

      // skip symbol table of classes, they can't be referenced by inner stuff.
      if( frame.m_type != CCFrame::t_func_type || &frame.m_elem.func->variables() == m_varmap )
      {
         // we can't close symbols across class definitions.
         break;
      }

      var = frame.m_elem.func->variables().find( name );
      if( var !=  0 )
      {
         if( var->type() == Variable::e_nt_local )
         {
            TRACE1("ParserContext::findSymbol \"%s\" found, need to be closed", name.c_ize() );
            //TODO: Properly close symbols. -- this won't work
            m_varmap->addClosed(name);
            return 0;
         }

         TRACE1("ParserContext::findSymbol \"%s\" found of type %d", name.c_ize(), var->type() );
         return var;
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
   m_varmap = el->varmap();
}

ExprLit* ParserContext::closeLitContext() 
{
   if( ! _p->m_litContexts.empty() )
   {
      ExprLit* current = _p->m_litContexts.back();
      _p->m_litContexts.pop_back();
      if( ! _p->m_litContexts.empty() ) {
         m_varmap = _p->m_litContexts.back()->varmap();
      }
      else if( !_p->m_frames.empty() ) {
         m_varmap = _p->m_frames.back().m_varmap;
      }
      else {
         m_varmap = 0;
      }

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


void ParserContext::openBlock( Statement* parent, SynTree* branch, bool bAutoClose )
{
   TRACE("ParserContext::openBlock type '%s'",
         parent->handler() == 0 ? "none": 
          parent->handler()->name().c_ize() );

   saveStatus( _p->m_frames.back() );

   //bool result = parent->discardable() ? true : checkSymbols();

   // if the pareser is not interactive, append the statement even after undefined errors.
   if( ! m_parser->hasErrors() || ! m_parser->interactive() )
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


void ParserContext::openFunc( SynFunc *func )
{
   TRACE("ParserContext::openFunc -- %s", func->name().c_ize() );

   saveStatus( _p->m_frames.back() );

   m_cfunc = func;
   m_st = &func->syntree();
   m_cstatement = 0;
   _p->m_frames.push_back(CCFrame(func));

   // get the symbol table.
   m_varmap = &func->variables();
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
   m_varmap = &fcls->makeConstructor()->variables();
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
         // is this a method?
         if ( m_cclass != 0 )
         {
            // unless it's the constructor -- in which case it's already added
            if ( bframe.m_elem.func->methodOf() != m_cclass )
            {
               m_cclass->addMethod( bframe.m_elem.func );
            }
         }
         else
         {
            onCloseFunc( bframe.m_elem.func );
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
   m_varmap = 0;
}

}

/* end of parsercontext.cpp */
