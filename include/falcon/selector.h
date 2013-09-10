/*
   FALCON - The Falcon Programming Language.
   FILE: selector.h

   VM Scheduler managing waits and sleeps of contexts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Feb 2013 20:56:12 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SELECTOR_H_
#define _FALCON_SELECTOR_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/shared.h>
#include <falcon/selectable.h>

namespace Falcon {

/**

 */
class FALCON_DYN_CLASS Selector: public Shared
{
public:
   Selector( ContextManager* mgr, const Class* handler=0, bool acquireable = false );

   /** Adds a new selectable entity to the selector, or modifies the status.
    * Mode is set additively.
    */

   void add( Selectable* stream, int mode, bool bAdditive = true );
   void update( Selectable* stream, int mode ) { add( stream, mode, false ); }
   void addRead( Selectable* stream );
   void addWrite( Selectable* stream );
   void addErr( Selectable* stream );

   static const int mode_read = 1;
   static const int mode_write = 2;
   static const int mode_err = 4;

   /** Removes a selectable entity from the next select requests.
    *
    * Even if removed, a stream with an incoming status
    * will still be retrieved.
    */
   bool remove( Selectable* stream );

   virtual void signal( int count = 1 );
   virtual int32 consumeSignal( VMContext* target, int32 count = 1 );
   virtual void gcMark( uint32 n );

   Selectable* getNextReadyRead();
   Selectable* getNextReadyWrite();
   Selectable* getNextReadyErr();

   void pushReadyRead( Selectable* stream );
   void pushReadyWrite( Selectable* stream );
   void pushReadyErr( Selectable* stream );

protected:
   virtual ~Selector();

   class Data {
   public:
      Selectable* m_resource;
      // how to read
      int m_mode;
      // true when posted to the waiter list.
      bool m_bPending;

      // multiplex where this data is subscribed
      Multiplex* m_mplex;

      Data():
         m_resource(0),
         m_mode(0),
         m_bPending(false)
      {}
   };

   virtual int lockedConsumeSignal( VMContext* target, int count = 1 );

private:
   void dequePending();
   void removeFromMultiplex( Selectable* found );

   class Private;
   Private* _p;

};

}

#endif

/* end of selector.h */

