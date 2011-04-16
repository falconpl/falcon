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


namespace Falcon {

//==================================================================
// Frame class
//

ParserContext::CCFrame::CCFrame():
   m_type( t_none_type )
{

}

ParserContext::CCFrame::CCFrame( Class* cls, bool bIsObject ):
   m_type( bIsObject ? t_object_type : t_class_type ),
   m_branch( 0 ),
   m_bStatePushed( false )
{
   m_elem.cls = cls;
}

ParserContext::CCFrame::CCFrame( SynFunc* func ):
   m_type( t_func_type ),
   m_branch( &func->syntree() ),
   m_bStatePushed( false )
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

//=============================================================

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


ParserContext::ParserContext( SourceParser *sp ):
   m_parser(sp),
   m_st(0),
   m_cstatement(0),
   m_cfunc(0),
   m_cclass(0)
{
   _p = new Private;
}

ParserContext::~ParserContext()
{
   delete _p;
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


void ParserContext::addStatement( Statement* stmt )
{
   TRACE("ParserContext::addStatement type %d", (int) stmt->type() );
   fassert( m_st != 0 );

   m_st->append(stmt);

   // TODO: make the unknown symbols something known!
   _p->m_unknown.clear();

   onNewStatement( stmt );
}

void ParserContext::openBlock( Statement* parent, SynTree* branch )
{
   TRACE("ParserContext::openBlock type %d", (int) parent->type() );
   m_cstatement = parent;
   m_st = branch;
   _p->m_frames.push_back( CCFrame(parent, branch) );
}

void ParserContext::changeBranch(SynTree* branch)
{
   TRACE("ParserContext::changeBranch", 0 );
   m_st = branch;
   _p->m_frames.back().m_branch = branch;
}

void ParserContext::openFunc( SynFunc *func )
{
   TRACE("ParserContext::openFunc -- %s", func->name().c_ize() );
   m_cfunc = func;
   m_cstatement = 0;
   _p->m_frames.push_back(CCFrame(func));
}

void ParserContext::openClass( Class *cls, bool bIsObject )
{
   TRACE("ParserContext::openClass -- depth %d %s%s", _p->m_frames.size() + 1,
            cls->name().c_ize(), bIsObject ? "":" (object)" );
   m_cclass = cls;
   m_cstatement = 0;
   _p->m_frames.push_back(CCFrame(cls, bIsObject));
}

void ParserContext::closeContext()
{
   TRACE("ParserContext::closeContext -- closing context on depth %d", _p->m_frames.size() );
   fassert( !_p->m_frames.empty() );
   
   // copy by value
   CCFrame bframe = _p->m_frames.back();

   // as we're removing the frame.
   _p->m_frames.pop_back();
   if( bframe.m_bStatePushed )
   {
      m_parser->popState();
   }

   // updating the syntactic tree
   if( !_p->m_frames.empty() )
   {
      m_st = _p->m_frames.back().m_branch;
   }

   // notify the new item.
   switch( bframe.m_type )
   {
      case CCFrame::t_none_type: fassert(0); break;

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
         onNewClass( bframe.m_elem.cls, bframe.m_type == CCFrame::t_object_type );
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
         onNewFunc( bframe.m_elem.func );
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

               if ( riter->m_type == CCFrame::t_func_type  )
               {
                  m_cstatement = riter->m_elem.stmt;
                  break;
               }

               ++riter;
            }
         }
         onNewStatement( bframe.m_elem.stmt );
         break;
   }
}

}

/* end of compcontext.cpp */

