/*
   FALCON - The Falcon Programming Language.
   FILE: exprmultiunpack.h

    Class handling a parallel array of expressions and symbols where to assign them
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Dec 2011 13:22:21 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRMULTIUNPACK_H_
#define FALCON_EXPRMULTIUNPACK_H_

#include <falcon/psteps/exprvector.h>

namespace Falcon {

/** Class handling a parallel array of expressions and symbols where to assign them. 

 */
class FALCON_DYN_CLASS ExprMultiUnpack: public ExprVector
{
public:
   ExprMultiUnpack( int line = 0, int chr = 0 );
   ExprMultiUnpack( const ExprMultiUnpack& other );
   virtual ~ExprMultiUnpack();

   inline virtual ExprMultiUnpack* clone() const { return new ExprMultiUnpack( *this ); }

   virtual bool simplify( Item& value ) const;
   virtual void render( TextWriter* tw, int32 depth ) const;
   inline virtual bool isStandAlone() const { return true; }
   virtual bool isStatic() const { return false; }

   // We accept assignments only
   virtual bool setNth( int32 n, TreeStep* ts );
   virtual bool insert( int32 pos, TreeStep* element );
   virtual bool append( TreeStep* element );

private:
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of exprmultiunpack.h */
