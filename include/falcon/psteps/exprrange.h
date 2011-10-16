/*
   FALCON - The Falcon Programming Language.
   FILE: exprrange.h

   Syntactic tree item definitions -- range generator.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 22 Sep 2011 13:26:43 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRRANGE_H
#define FALCON_EXPRRANGE_H

#include <falcon/expression.h>

namespace Falcon
{

/** Expression generating a range instance.
 */
class FALCON_DYN_CLASS ExprRange: public Expression
{
public:
   ExprRange();
   ExprRange( Expression *estart, Expression* eend=0, Expression* estep=0 );
   ExprRange( const ExprRange& other );
   virtual ~ExprRange();
   
   Expression* start() const { return m_estart; }
   Expression* end() const { return m_eend; }
   Expression* step() const { return m_estep; }

   void start( Expression* expr );
   void end( Expression* expr );
   void step( Expression* expr );
   
   virtual void serialize( DataWriter* s ) const;
   virtual void precompile( PCode* pcd ) const;
   
   static void apply_( const PStep*, VMContext* ctx );

   inline virtual ExprRange* clone() const { return new ExprRange( *this ); }
   virtual bool isStatic() const { return false; }
   virtual bool simplify( Item& result ) const;

protected:
   virtual void deserialize( DataReader* s );
   
private:
   Expression* m_estart;
   Expression* m_eend;
   Expression* m_estep;
};

}

#endif

/* end of exprrange.h */
