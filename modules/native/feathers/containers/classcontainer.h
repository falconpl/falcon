/*********************************************************************
 * FALCON - The Falcon Programming Language.
 * FILE: classcontainer.h
 *
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sat, 01 Feb 2014 12:56:12 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: The above AUTHOR
 *
 * See LICENSE file for licensing details.
 */

#ifndef _FALCON_FEATHERS_CONTAINERS_CLASSCONTAINER_H_
#define _FALCON_FEATHERS_CONTAINERS_CLASSCONTAINER_H_

#include <falcon/class.h>

namespace Falcon {

class PStep;

/** Base class for the handlers of a container.
 *
 */
class ClassContainerBase: public Class
{
public:
   ClassContainerBase( const String& name );
   virtual ~ClassContainerBase();

   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream ) const;
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;

   virtual void op_isTrue( VMContext* ctx, void* instance ) const;
   virtual void op_in( VMContext* ctx, void* instance ) const;
   virtual void op_iter( VMContext* ctx, void* instance ) const;
   virtual void op_next( VMContext* ctx, void* instance ) const;


   /** This step checks if an item is in a sequence.
    * Signature: (2: item to be searched) (1: ClassIteartor|iterator) (0: dummy int 1) --> (0: boolean)
    */
   PStep* stepContains() const { return m_stepContains; }

private:
   PStep* m_stepContains;
};


class ClassContainer: public ClassContainerBase
{
public:
   ClassContainer();
   virtual ~ClassContainer();
   virtual void* createInstance() const;
};

}

#endif

/* end of classcontainer.h */
