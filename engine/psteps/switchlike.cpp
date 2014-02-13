/*
   FALCON - The Falcon Programming Language.
   FILE: switchlike.cpp

   Parser for Falcon source files -- Switch and select base classes.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/switchlike.cpp"

#include <falcon/syntree.h>
#include <falcon/expression.h>
#include <falcon/textwriter.h>

#include <falcon/psteps/switchlike.h>
#include <falcon/psteps/exprcase.h>

#include <vector>

namespace Falcon {

class SwitchlikeStatement::Private
{
public:
   typedef std::vector< SynTree* > Blocks;
   Blocks m_blocks;

   Private() {}

   Private( SwitchlikeStatement* owner, const Private& other )
   {
      m_blocks.reserve(other.m_blocks.size());

      Blocks::const_iterator cbi = other.m_blocks.begin();
      while( cbi != other.m_blocks.end() )
      {
         SynTree* ob = *cbi;
         SynTree* st = ob->clone();
         st->setParent(owner);
         m_blocks.push_back( st );
         ++cbi;
      }
   }

   ~Private()
   {
      Blocks::iterator iter = m_blocks.begin();
      while( iter != m_blocks.end() )
      {
         dispose( *iter );
         ++iter;
      }
   }
};


SwitchlikeStatement::SwitchlikeStatement( int32 line, int32 chr ):
   Statement( line, chr ),
   m_expr( 0 ),
   m_dummyTree(0)
{
   _p = new Private;
}

SwitchlikeStatement::SwitchlikeStatement( Expression* expr, int32 line, int32 chr ):
   Statement( line, chr ),
   m_expr( expr ),
   m_dummyTree(0)
{
   expr->setParent(this);
   _p = new Private;
}

SwitchlikeStatement::SwitchlikeStatement( const SwitchlikeStatement& other ):
   Statement( other ),
   m_expr(0),
   m_dummyTree(0)
{
   if( other.m_expr != 0 )
   {
      m_expr = other.m_expr->clone();
      m_expr->setParent(this);
   }

   _p = new Private(this, *other._p);
}


SwitchlikeStatement::~SwitchlikeStatement()
{
   delete _p;
   dispose( m_dummyTree );
   dispose( m_expr );
}



int32 SwitchlikeStatement::arity() const
{
   return (int) _p->m_blocks.size();
}


TreeStep* SwitchlikeStatement::nth( int32 n ) const
{
   int32 size = (int32) _p->m_blocks.size();
   if( n < 0 ) n = size + n;
   if( n < 0 || n >= size ) return 0;

   return _p->m_blocks[n];
}


bool SwitchlikeStatement::setNth( int32 n, TreeStep* ts )
{
   int32 size = (int32) _p->m_blocks.size();

   if( n == size )
   {
      return append( ts );
   }

   if( n < 0 ) n = size + n;
   if( n < 0 || n > size
            || ts->category() != TreeStep::e_cat_syntree
            || (ts->selector() == 0 || ts->category() != TreeStep::e_cat_expression ||
                     static_cast<Expression*>(ts->selector())->trait() != Expression::e_trait_case )
            || ! ts->setParent(this) )
   {
      return false;
   }

   dispose( _p->m_blocks[n] );
   _p->m_blocks[n] = static_cast<SynTree*>(ts);
   return true;
}


bool SwitchlikeStatement::insert( int32 pos, TreeStep* ts )
{
   int32 size = (int32) _p->m_blocks.size();

   if( pos == size )
   {
      return append( ts );
   }

   if( pos < 0 ) pos = size + pos;
   if( pos < 0 || pos > size
            || ts->category() != TreeStep::e_cat_syntree
            || (ts->selector() == 0 || ts->category() != TreeStep::e_cat_expression ||
                     static_cast<Expression*>(ts->selector())->trait() != Expression::e_trait_case )
            || ! ts->setParent(this) )
   {
      return false;
   }

   _p->m_blocks.insert( _p->m_blocks.begin() + pos, static_cast<SynTree*>(ts) );
   return true;
}


bool SwitchlikeStatement::append( TreeStep* element )
{
   int32 size = (int32) _p->m_blocks.size();
   if( element->category() != TreeStep::e_cat_syntree )
   {
      return false;
   }

   SynTree* st = static_cast<SynTree*>(element);

   if( element->selector() == 0 )
   {
      if( ! element->setParent( this ) )
      {
         return false;
      }

      // no previous default? -- this is the new one.
      _p->m_blocks.insert( _p->m_blocks.end(), st );
   }
   else
   {
      if( element->selector()->category() != TreeStep::e_cat_expression ||
               static_cast<Expression*>(element->selector())->trait() != Expression::e_trait_case || ! element->setParent( this ) )
      {
         return false;
      }

      // shift default down?
      if( size > 0 && _p->m_blocks[size-1]->selector() == 0 )
      {
         _p->m_blocks.insert( _p->m_blocks.begin() + (size-1), st );
      }
      else {
         // no default? -- insert at end.
         _p->m_blocks.insert( _p->m_blocks.end(), st );
      }
   }

   return true;
}


bool SwitchlikeStatement::remove( int32 pos )
{
   int32 size = (int32) _p->m_blocks.size();

   if( pos < 0 ) pos = size + pos;
   if( pos < 0 || pos >= size )
   {
      return false;
   }

   dispose( _p->m_blocks[pos] );
   _p->m_blocks.erase( _p->m_blocks.begin() + pos );
   return true;
}


SynTree* SwitchlikeStatement::dummyTree()
{
   if( m_dummyTree == 0 ) {
      m_dummyTree = new SynTree();
      m_dummyTree->setParent(this);
   }
   
   return m_dummyTree;
}


template<class __TI, class __T>
SynTree* SwitchlikeStatement::findBlock( const __TI& value, const __T& verifier ) const
{
   Private::Blocks::iterator iter = _p->m_blocks.begin();
   while( iter != _p->m_blocks.end() )
   {
      SynTree* st = *iter;
      TreeStep* sel = st->selector();
      if( sel == 0 )
      {
         // the default block
         return st;
      }

      if( sel->category() == TreeStep::e_cat_expression && static_cast<Expression*>(sel)->trait() == Expression::e_trait_case )
      {
         ExprCase* cs = static_cast<ExprCase*>(sel);
         if( verifier.check(cs, value) ) {
            return st;
         }
      }
      ++iter;
   }

   return 0;
}


class ValueVerifier
{
public:
   bool check( ExprCase* cs, const Item& value ) const
   {
      return cs->verify(value);
   }
};

class SymbolVerifier
{
public:
   bool check( ExprCase* cs, const Symbol* value ) const
   {
      return cs->verifySymbol(value);
   }
};



class TypeVerifier
{
public:
   TypeVerifier( VMContext* ctx ):
      m_ctx(ctx)
   {}

   bool check( ExprCase* cs, const Item& value ) const
   {
      return cs->verifyType(value, m_ctx);
   }

private:
   VMContext* m_ctx;
};


class LiveValueVerifier
{
public:
   LiveValueVerifier( VMContext* ctx ):
      m_ctx(ctx)
   {}

   bool check( ExprCase* cs, const Item& value ) const
   {
      return cs->verifyItem(value, m_ctx);
   }

private:
   VMContext* m_ctx;
};



SynTree* SwitchlikeStatement::findBlockForItem( const Item& value ) const
{
   ValueVerifier vv;
   return findBlock(value, vv);
}


SynTree* SwitchlikeStatement::findBlockForSymbol( const Symbol* value ) const
{
   SymbolVerifier vv;
   return findBlock(value, vv);
}

SynTree* SwitchlikeStatement::findBlockForItem( const Item& value, VMContext* ctx ) const
{
   LiveValueVerifier vv(ctx);
   return findBlock(value, vv);
}

SynTree* SwitchlikeStatement::findBlockForType( const Item& value, VMContext* ctx ) const
{
   TypeVerifier v(ctx);
   return findBlock(value, v );
}

SynTree* SwitchlikeStatement::getDefault() const
{
   if( _p->m_blocks.size() == 0 )
   {
      return 0;
   }
   if( _p->m_blocks.back()->selector() == 0 )
   {
      return _p->m_blocks.back();
   }

   return 0;
}

/** Returns the selector for this expression.*/
TreeStep* SwitchlikeStatement::selector() const
{
   return m_expr;
}


bool SwitchlikeStatement::selector( TreeStep* expr )
{
   if( expr == 0 )
   {
      return false;
   }

   if( ! expr->setParent(this) )
   {
      return false;
   }

   dispose(m_expr);
   m_expr = expr;

   return true;
}


void SwitchlikeStatement::render( TextWriter* tw, int32 depth ) const
{
   if( selector() == 0 )
   {
      tw->write( renderPrefix(depth) );
      tw->write( "/* Blank switch/select */" );
   }
   else
   {
      int32 dp = depth < 0 ? -depth : depth+1;

      renderHeader(tw, depth);
      Private::Blocks::iterator iter = _p->m_blocks.begin();
      while( iter != _p->m_blocks.end() )
      {
         SynTree* st = *iter;
         TreeStep* sel = st->selector();
         if( sel == 0 || sel->category() != TreeStep::e_cat_expression || static_cast<Expression*>(sel)->trait() != Expression::e_trait_case )
         {
            tw->write( renderPrefix(dp) );
            tw->write( "default\n" );
         }
         else {
            tw->write( renderPrefix(dp) );
            tw->write("case ");
            sel->render(tw, relativeDepth(dp));
            tw->write( "\n" );
         }

         st->render( tw, dp + 1 );
         ++iter;
         if( sel == 0 || sel->category() != TreeStep::e_cat_expression || static_cast<Expression*>(sel)->trait() != Expression::e_trait_case )
         {
            // force the default block to be the last
            break;
         }
      }

      tw->write( renderPrefix(depth) );
      tw->write("end");
   }

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}


}

/* end of switchlike.cpp */
