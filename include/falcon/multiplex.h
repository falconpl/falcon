/*
   FALCON - The Falcon Programming Language.
   FILE: multiplex.h

   Multiplex framework for Streams and Selectors.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 28 Feb 2013 18:18:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_MULTIPLEX_H_
#define _FALCON_MULTIPLEX_H_

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon
{

class Stream;
class Selector;

/**
 * Stream multiplexer.
 *
 * A class that can multiplex stream instances polling for their
 * availability status.
 *
 * This class is usually not created directly; the MultiplexGenerator::generate()
 * method is used to produce new instances of this class.
 *
 * \note The multiplex instances are owned by the selector where they
 * are instantiated.
 *
 * \see getMultiplexGenerator();
 */
class Multiplex
{
public:
   virtual ~Multiplex(){}

   virtual void gcMark( uint32 mark );
   uint32 currentMark() const { return m_mark; }

   /** Adds a stream to the multiplexing.
    *
    * \param Stream stream to be added (or modified).
    * \param mode A read/write/error poll mode as bitfield according to Selector::mode_*
    *
    */
   virtual void addStream( Stream* stream, int mode ) = 0;

   /** Removes a stream from multiplexing (as soon as possible).
    *
    * \param Stream stream to be removed.
    *
    * The subclass should remove the multiplexed entity as soon as possible,
    * but it is allowed to send this same stream to a ready queue in the
    * selector after this call. So, in case it's difficult or impractical to
    * remove this stream from the selection right now, this can be delayed to
    * a future moment.
    *
    * \param Management of refcounts for streams is up to the multiplex instance;
    * the selector has its own counts.
    */
   virtual void removeStream( Stream* stream ) = 0 ;

   MultiplexGenerator* generator() const { return m_generator; }

   Selector* selector() const { return m_selector; }

protected:
   /** Creates the multiplex on a selector.
    * The module is used for back-reference and keep alive marks.
    */
   Multiplex( Selector* master,  Module* mod = 0 ):
      m_generator(generator),
      m_selector(master),
      m_module(mod),
      m_mark(0)
   {}

   /**
    * Invoked when a stream is ready for read.
    * \param stream The stream now ready for read.
    *
    * Call this back when a stream is found ready for read.
    * \note This function increments the reference count for the stream;
    * the count is decremented as the stream is dequeued by the Selector's user.
    */
   void onReadyRead( Stream* stream );

   /**
    * Invoked when a stream is ready for write.
    * \param stream The stream now ready for write.
    *
    * Call this back when a stream is found ready for write.
    * \note This function increments the reference count for the stream;
    * the count is decremented as the stream is dequeued by the Selector's user.
    */
   void onReadyWrite( Stream* stream );

   /**
    * Invoked when a stream is found having an error signal active.
    * \param stream The stream has an error signal active.
    *
    * Call this back when a stream line has an error, high priority
    * or out of band signal active.
    *
    * \note This function increments the reference count for the stream;
    * the count is decremented as the stream is dequeued by the Selector's user.
    */
   void onReadyErr( Stream* stream );

   MultiplexGenerator* m_generator;
   Selector* m_selector;
   Module* m_module;
   uint32 m_mark;
};

}

#endif /* MULTIPLEX_H_ */

/* end of multiplex.h */
