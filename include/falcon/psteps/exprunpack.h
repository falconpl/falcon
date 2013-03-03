/*
   FALCON - The Falcon Programming Language.
   FILE: exprunpack.h

   Expression used to unpack a single value into multiple symbols
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 02 Jun 2011 13:39:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRUNPACK_H_
#define FALCON_EXPRUNPACK_H_

#include <falcon/expression.h>

#include <vector>

namespace Falcon {

/** Expression used to unpack a single value into multiple symbols. */
class FALCON_DYN_CLASS ExprUnpack: public Expression
{
public:
   ExprUnpack( int line = 0, int chr = 0 );
   ExprUnpack( Expression* op1, int line = 0, int chr = 0 );
   ExprUnpack( const ExprUnpack& other );
   virtual ~ExprUnpack();

   inline virtual ExprUnpack* clone() const { return new ExprUnpack( *this ); }
   virtual bool simplify( Item& value ) const;
   virtual void describeTo( String&, int depth = 0 ) const;
   inline virtual bool isStandAlone() const { return true; }

   int targetCount() const;
   Symbol* getAssignand( int n ) const;
   ExprUnpack& addAssignand( Symbol* );

   virtual bool isStatic() const { return false; }

   virtual Expression* selector() const
   {
      return m_expander;
   }

   bool selector( Expression* sel );

protected:
   Expression* m_expander;
   
private:
   class Private;
   Private* _p;

   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif 

/* end of exprunpack.cpp */
