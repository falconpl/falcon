/*
   FALCON - The Falcon Programming Language.
   FILE: classbool.h

   Class defined by a Falcon script.
   -------------------------------------------------------------------
   Author: Francesco Magliocca
   Begin: Sun, 19 Jun 2011 12:40:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSBOOL_H_
#define _FALCON_CLASSBOOL_H_

#include <falcon/setup.h>
#include <falcon/class.h>
#include <falcon/pstep.h>

namespace Falcon
{

/** Class defined by a Falcon script.

 This class implements a class handler for classes a Falcon script. In other words,
 it is a handler for the "class type". The content of this type is a FalconClass,
 where properties and methods declared in a Falcon script class declaration
 are stored.
 
 */
class FALCON_DYN_CLASS ClassBool : public Class
{
public:

   ClassBool();
   virtual ~ClassBool();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void store( VMContext*, DataWriter* , void* ) const;
   virtual void restore( VMContext* , DataReader* ) const;
   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;

   //=============================================================

   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   virtual void op_isTrue( VMContext* ctx, void* self ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;

private:

   class FALCON_DYN_CLASS NextOpCreate: public PStep {
   public:
      NextOpCreate() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
   } m_OP_create_next;
};

}

#endif /* _FALCON_CLASSBOOL_H_ */

/* end of coreclass.h */
