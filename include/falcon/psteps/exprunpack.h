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

#include <falcon/psteps/exprvector.h>

#include <vector>

namespace Falcon {

class PStepAssignAllValues;

/** Expression used to unpack a single value into multiple symbols. */
class FALCON_DYN_CLASS ExprUnpack: public ExprVector
{
public:
   ExprUnpack( int line = 0, int chr = 0 );
   ExprUnpack( Expression* op1, int line = 0, int chr = 0 );
   ExprUnpack( const ExprUnpack& other );
   virtual ~ExprUnpack();

   inline virtual ExprUnpack* clone() const { return new ExprUnpack( *this ); }
   virtual bool simplify( Item& value ) const;
   virtual void render( TextWriter* tw, int32 depth ) const;
   inline virtual bool isStandAlone() const { return true; }

   virtual TreeStep* selector() const
   {
      return m_expander;
   }

   bool selector( TreeStep* sel );

   virtual bool setNth( int32 n, TreeStep* ts );
   virtual bool insert( int32 pos, TreeStep* element );
   virtual bool append( TreeStep* element );

protected:
   TreeStep* m_expander;
   
private:
   friend class PStepAssignAllValues;

   static void apply_( const PStep*, VMContext* ctx );
   PStep* m_stepAssignAllValues;
};

}

#endif 

/* end of exprunpack.cpp */
