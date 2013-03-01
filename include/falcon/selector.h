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

namespace Falcon {

class Stream;

/**

 */
class FALCON_DYN_CLASS Selector: public Shared
{
public:
   Selector( ContextManager* mgr, const Class* handler=0, bool acquireable = false );

   /** Adds a new stream to the selector, or modifies the status.
    * Mode is set additively.
    */

   void add( Stream* stream, int mode, bool bAdditive = true );
   void update( Stream* stream, int mode ) { add( stream, mode, false ); }
   void addRead( Stream* stream );
   void addWrite( Stream* stream );
   void addErr( Stream* stream );

   static const int mode_read = 1;
   static const int mode_write = 2;
   static const int mode_err = 4;

   /** Removes a stream from the next select requests.
    *
    * Even if removed, a stream with an incoming status
    * will still be retrieved.
    */
   bool remove( Stream* stream );

   virtual int32 consumeSignal( VMContext* target, int32 count = 1 );
   virtual void gcMark( uint32 n );

   Stream* getNextReadyRead();
   Stream* getNextReadyWrite();
   Stream* getNextReadyErr();

   void pushReadyRead( Stream* stream );
   void pushReadyWrite( Stream* stream );
   void pushReadyErr( Stream* stream );

protected:
   virtual ~Selector();

   class Data {
   public:
      Stream* m_stream;
      // how to read
      int m_mode;
      // true when posted to the waiter list.
      bool m_bPending;

      Data():
         m_stream(0),
         m_mode(0),
         m_bPending(false)
      {}
   };

   virtual int32 lockedConsumeSignal( VMContext* target, int32 count = 1 );

private:
   void dequePending();
   void removeFromMultiplex( Stream* found );

   class Private;
   Private* _p;

};

}

#endif

/* end of selector.h */

