/*
   FALCON - The Falcon Programming Language.
   FILE: exprvector_private.h

   Inner structure holding a vector of expressions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 18:38:19 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRVECTOR_PRIVATE_H
#define FALCON_EXPRVECTOR_PRIVATE_H

#include <falcon/expression.h>
#include <vector>

namespace Falcon
{

template <class TreeStepSubClass__>
class TSVector_Private {
public:

   typedef std::vector< TreeStepSubClass__* > ExprVector;
   typedef typename ExprVector::iterator ExprVector_i;
   typedef typename ExprVector::const_iterator ExprVector_ci;
   ExprVector m_exprs;
   
   inline TSVector_Private() {}
   
   inline TSVector_Private( const TSVector_Private<TreeStepSubClass__>& other, TreeStep* owner ) {
      const ExprVector& oe = other.m_exprs;
      ExprVector& mye = m_exprs;

      mye.reserve(oe.size());
      ExprVector_ci iter = oe.begin();
      while( iter != oe.end() )
      {
         TreeStepSubClass__* expr = static_cast<TreeStepSubClass__*>((*iter)->clone());
         expr->setParent( owner );
         mye.push_back( expr );
         
         ++iter;
      }
   }
   
   
   ~TSVector_Private()
   {
      clear();
   }

   void clear()
   {
      ExprVector_i iter = m_exprs.begin();
      while( iter != m_exprs.end() )
      {
         TreeStep* child = *iter;
         TreeStep::dispose( child );
         ++iter;
      }
      m_exprs.clear();
   }
   
   inline int arity() const
   {
      return (int) m_exprs.size();
   }

   inline TreeStepSubClass__* nth( int32 n ) const
   {
      if( n < 0 ) n = (int) m_exprs.size() + n;
      if( n < 0 || n >= (int) m_exprs.size() ) return 0;

      return m_exprs[n];
   }

   bool nth( int32 n, TreeStepSubClass__* ts, TreeStep* owner)
   {
      if( n < 0 ) n = (int) m_exprs.size() + n;
      if( n < 0 || n > (int) m_exprs.size() ) return false;
      if( ts != 0 && ! ts->setParent(owner) ) return false;
      
      if( n == (int) m_exprs.size() ) {
         m_exprs.push_back(ts);
      }
      else {
         delete m_exprs[n];
         m_exprs[n] = ts;
      }
      return true;
   }

   inline bool insert( int32 n, TreeStepSubClass__* ts, TreeStep* owner )
   {
      if( ts != 0 && ! ts->setParent(owner) ) return false;

      if( n < 0 ) n = (int) m_exprs.size() + n;

      if( n < 0 || n >= (int) m_exprs.size() ) {      
         m_exprs.push_back(ts);
      }
      else {
         m_exprs.insert( m_exprs.begin() + n, ts);
      }

      return true;
   }

   inline bool remove( int32 n )
   {   
      if( n < 0 ) n = (int) m_exprs.size() + n;

      if( n < 0 || n >= (int) m_exprs.size() ) {      
         return false;
      }

      delete m_exprs[n];
      m_exprs.erase( m_exprs.begin() + n );
      return true;
   }
   
   inline bool append( TreeStepSubClass__* ts, TreeStep* owner )
   {      
      if( ts != 0 && ! ts->setParent(owner) ) return false;  
      m_exprs.push_back(ts);
      return true;
   }
};


class TreeStepVector_Private: public TSVector_Private<TreeStep>
{
public:

   TreeStepVector_Private() {}
   ~TreeStepVector_Private() {}

   TreeStepVector_Private( const TreeStepVector_Private& other, TreeStep* owner ):
      TSVector_Private<TreeStep>( other, owner )
   {}
};



}

#endif

/* end of exprvector_private.h */
