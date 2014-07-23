/*
   FALCON - The Falcon Programming Language.
   FILE: classeventcourier.h

   Sript/VM interface to the EventCourier message dispatching system
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 15 Jul 2014 16:48:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CLASSEVENTCOURIER_H
#define FALCON_CLASSEVENTCOURIER_H

#include <falcon/class.h>

namespace Falcon
{
class Function;
class PStep;

/** Sript/VM interface to the EventCourier message dispatching system
 *
 */
class FALCON_DYN_CLASS ClassEventCourier: public Class
{
public:
   ClassEventCourier();
   virtual ~ClassEventCourier();

   /// Called by the engine to configure the class post creation.
   void init();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   Function* waitFunction() const { return m_funcWait; }
   const PStep* stepEngage() const { return m_stepEngage; }
   const PStep* stepAfterWait() const { return m_stepAfterWait; }
   const PStep* stepAfterHandling() const { return m_stepAfterHandling; }
   const PStep* stepAfterHandlingCatch() const { return m_stepAfterHandlingCatch; }
   const PStep* stepAfterSendWait() const { return m_stepAfterSendWait; }
   const Class* tokenClass() const { return m_tokenClass; }

private:

   Function* m_funcWait;
   PStep* m_stepEngage;
   PStep* m_stepAfterWait;
   PStep* m_stepAfterHandling;
   PStep* m_stepAfterHandlingCatch;
   PStep* m_stepAfterSendWait;

   Class* m_tokenClass;
};

}

#endif

/* classeventcourier.h */
