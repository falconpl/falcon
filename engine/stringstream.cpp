/*
   FALCON - The Falcon Programming Language.
   FILE: stringstream.cpp

   Implementation of StringStream oriented streams.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Apr 2011 21:57:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/stringstream.cpp"

#include <falcon/stringstream.h>
#include <falcon/selector.h>
#include <falcon/stdhandlers.h>
#include <falcon/stdmpxfactories.h>

#include <cstring>
#include <string.h>

#include <stdio.h>

#include <set>

#define STRING_STREAM_STACK_SIZE (32*1024)

namespace Falcon {

/*
 A strange muiltiplex: we'll have actually many of these for a single StringStream.
 */
class StringStream::MPX: public Multiplex
{
public:
   MPX( const Multiplex::Factory* generator, Selector* master );
   virtual ~MPX();

   virtual void add( Selectable* stream, int mode );
   virtual void remove( Selectable* stream );
   virtual uint32 size() const { return m_size; }

   void onStringStreamReady( StringStream* ss );

private:
   FALCON_REFERENCECOUNT_DECLARE_INCDEC(MPX);
   uint32 m_size;
};



class StringStream::Buffer {
public:
   String* m_str;
   atomic_int m_refcount;
   Mutex m_mtx;
   
   typedef std::set<StringStream::MPX*> WaiterSet;
   WaiterSet m_waiters;

   Buffer():
      m_str( new String() ),
      m_refcount(1)
   {}

   Buffer( const Buffer& other ):
      m_str( new String( *other.m_str ) ),
      m_refcount(1)
   {}

   void incref() { atomicInc(m_refcount); }
   void decref() { if (atomicDec(m_refcount) == 0) delete this; }

private:
   ~Buffer() {
      delete m_str;
   }
};


StringStream::StringStream( int32 size ):
   m_posRead(0),
   m_posWrite(0),
   m_bPipeMode(false),
   m_b( new Buffer ),
   m_selectable(0)
{
   if ( size <= 0 )
      size = 32;

   m_b->m_str->reserve(size);
   status(t_open);
}


StringStream::StringStream( const String &strbuf ):
   m_posRead(0),
   m_posWrite(0),
   m_bPipeMode(false),
   m_b( new Buffer ),
   m_selectable(0)
{
   *m_b->m_str = strbuf;
   m_b->m_str->bufferize();
   status(t_open);
}

StringStream::StringStream( byte* data, int64 size ):
   m_posRead(0),
   m_posWrite(0),
   m_bPipeMode(false),
   m_b( new Buffer ),
   m_selectable(0)
{
   m_b->m_str->adoptMemBuf(data, size, size );
}

StringStream::StringStream( const StringStream &strbuf ):
   Stream( strbuf ),
   m_posRead( strbuf.m_posRead ),
   m_posWrite( strbuf.m_posWrite ),
   m_bPipeMode(strbuf.m_bPipeMode),
   m_selectable(0)
{
   strbuf.m_b->incref();
   m_b = strbuf.m_b;
   status(t_open);
}


StringStream::~StringStream()
{
   m_b->decref();
   // we can't be destroyed with a selectable active
   fassert( m_selectable == 0 );
}


void StringStream::setBuffer( const String &source )
{
   m_posRead = 0;
   m_posWrite = 0;
   
   m_b->m_mtx.lock();
   if( m_b->m_str == 0 )
   {
      m_b->m_str = new String( source );
   }
   else
   {
      *m_b->m_str = source;
   }

   m_b->m_str->bufferize();
   m_lastError = 0;
   m_b->m_mtx.unlock();
}


void StringStream::setBuffer( const char* source, int size )
{
   m_b->m_mtx.lock();
   m_posRead = 0;
   m_posWrite = 0;

   if( m_b->m_str == 0 )
   {
      m_b->m_str = new String;
   }
   m_b->m_str->reserve(size);
   memcpy( m_b->m_str->getRawStorage(), source, size );
   m_b->m_str->size( size );
   m_b->m_str->manipulator( &csh::handler_buffer );
   m_lastError = 0;
   m_b->m_mtx.unlock();
}


bool StringStream::detachBuffer()
{
   m_b->m_mtx.lock();
   if( m_b->m_str != 0 ) {
      m_b->m_str->setRawStorage(0);
      m_b->m_str->size(0);
      m_b->m_str->allocated(0);
      m_b->m_str->manipulator( &csh::handler_static );
      m_b->m_mtx.unlock();
      
      status( t_none );
      return true;
   }
   
   m_b->m_mtx.unlock();
   return false;
}

uint32 StringStream::length() const 
{ 
   return m_b->m_str->size();
}

uint32 StringStream::allocated() const 
{ 
   return m_b->m_str->allocated();
}

byte *StringStream::data() const
{ 
   return m_b->m_str->getRawStorage();
}


Class* StringStream::handler()
{
   return Engine::handlers()->stringStreamClass();
}


String *StringStream::closeToString()
{
   m_b->m_mtx.lock();
   String *s = m_b->m_str;
   m_b->m_str = 0;
   m_b->m_mtx.unlock();
   return s;
}

void StringStream::transferFrom( StringStream &strbuf )
{
   strbuf.m_b->incref();
   m_b->decref();
   m_b = strbuf.m_b;
}


bool StringStream::close()
{
   m_b->m_mtx.lock();
   delete m_b->m_str;
   m_b->m_str = 0;
   m_b->m_mtx.unlock();
   return false;
}

size_t StringStream::read( void *buffer, size_t size )
{
   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 ) {
      m_status = t_error;
      m_b->m_mtx.unlock();

      return -1;
   }

   uint32 bsize =  m_b->m_str->size();

   if ( m_posRead >= bsize ) {
      m_b->m_mtx.unlock();
      
      return 0;
   }

   int sret = (int)(size + m_posRead < bsize ? size : bsize - m_posRead);
   memcpy( buffer, m_b->m_str->getRawStorage() + m_posRead, sret );
   m_posRead += sret;

   if(! m_bPipeMode )
   {
      m_posWrite = m_posRead;
      if ( m_posRead >= bsize ) {
         m_status = m_status | t_eof;
      }
   }

   m_b->m_mtx.unlock();

   return sret;
}


size_t StringStream::write( const void *buffer, size_t size )
{
   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return -1;
   }
   
   // be sure there is enough space to write
   m_b->m_str->reserve((length_t) m_posWrite+size);

   // effectively write
   memcpy( m_b->m_str->getRawStorage() + m_posWrite, buffer, size );
   m_posWrite += size;

   // are we writing at end? -- enlarge the string.
   if( m_posWrite > m_b->m_str->size() )
   {
      m_b->m_str->size( (length_t) m_posWrite );
      m_status = m_status | Stream::t_eof;
   }
   else {
      m_status = m_status & ~Stream::t_eof;
   }

   if(! m_bPipeMode )
   {
      m_posRead = m_posWrite;
   }

   if( m_posRead < m_b->m_str->size() )
   {
      // inform all the waiters
      Buffer::WaiterSet::iterator iter = m_b->m_waiters.begin();
      while( iter != m_b->m_waiters.end() )
      {
         MPX* mpx = *iter;
         mpx->onStringStreamReady(this);
         ++iter;
      }
      m_b->m_waiters.clear();
   }

   m_b->m_mtx.unlock();

   return size;
}


int64 StringStream::seek( int64 pos, Stream::e_whence w )
{
   m_b->m_mtx.lock();

   if ( m_b->m_str == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return -1;
   }

   switch( w ) {
      case Stream::ew_begin: m_posWrite = (int32) pos; break;
      case Stream::ew_cur: m_posWrite += (int32) pos; m_posRead= m_posWrite; break;
      case Stream::ew_end: m_posWrite = (int32) (m_b->m_str->size() + (pos-1)); break;
   }

   if ( m_posWrite > m_b->m_str->size() )
   {
      m_posWrite = m_b->m_str->size();
      m_status = t_eof;
   }
   else if ( m_posWrite < 0LL )
   {
      m_posWrite = 0;
      m_status = m_status & ~t_eof;
   }
   else
   {
      m_status = m_status & ~t_eof;
   }

   pos = m_posRead = m_posWrite;
   m_b->m_mtx.unlock();

   return pos;
}

int64 StringStream::tell()
{
   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return -1;
   }
   
   int64 pos = m_posRead;
   m_b->m_mtx.unlock();
   return pos;
}

bool StringStream::truncate( int64 pos )
{
   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 ) {
      m_b->m_mtx.unlock();
      m_status = t_error;
      return false;
   }

   if ( pos <= 0 )
      m_b->m_str->size( 0 );
   else
      m_b->m_str->size( (int32) pos );

   m_b->m_mtx.unlock();
   return true;
}

void StringStream::getString( String &target ) const
{
   m_b->m_mtx.lock();
   // as the string is always buffered, this operation is safe
   target = *m_b->m_str;
   m_b->m_mtx.unlock();
}

String StringStream::getStringAndClear()
{
	m_b->m_mtx.lock();
	
	String s = *m_b->m_str;
	
	m_b->m_str->size( 0 );
	
	m_posRead = 0;
    m_posWrite = 0;
	
	m_b->m_mtx.unlock();
	
	return s;
}

bool StringStream::closeToString( String &target )
{
   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 )
   {
      m_b->m_mtx.unlock();
      return false;
   }
   
   // adopt original string buffer.
   target.adopt( (char*)m_b->m_str->getRawStorage(), m_b->m_str->size(), m_b->m_str->allocated() );
   target.manipulator( (csh::Base*) m_b->m_str->manipulator() ); // apply correct manipulator


   // dispose the old string without disposing of its memory
   m_b->m_str->allocated( 0 );
   m_b->m_str->setRawStorage( 0 );
   delete m_b->m_str;
   m_b->m_str = 0;
   m_b->m_mtx.unlock();

   return true;
}



void StringStream::setPipeMode( bool mode )
{
   m_b->m_mtx.lock();
   m_bPipeMode = mode;
   m_bPS = mode;
   if( ! m_bPipeMode )
   {
      m_posWrite = m_posRead;
   }
   m_b->m_mtx.unlock();
}


bool StringStream::isPipeMode() const
{
   m_b->m_mtx.lock();
   bool mode = m_bPipeMode;
   m_b->m_mtx.unlock();

   return mode;
}


byte * StringStream::closeToBuffer()
{
   m_b->m_mtx.lock();
   if ( m_b->m_str == 0 || m_b->m_str->size() == 0)
   {
      m_b->m_mtx.unlock();
      return 0;
   }
   byte *retbuf = m_b->m_str->getRawStorage();

   // dispose the old string without disposing of its memory
   m_b->m_str->allocated( 0 );
   m_b->m_str->setRawStorage( 0 );
   delete m_b->m_str;
   m_b->m_str = 0;

   m_b->m_mtx.unlock();
   
   return retbuf;
}


StringStream *StringStream::clone() const
{
   StringStream *sstr = new StringStream( *this );
   return sstr;
}


const Multiplex::Factory* StringStream::multiplexFactory() const
{
   static const Multiplex::Factory* gen = Engine::mpxFactories()->stringStreamMpxFact();
   return gen;
}



StringStream::MpxFactory::~MpxFactory()
{}

Multiplex* StringStream::MpxFactory::create( Selector* master ) const
{
   return new MPX(this, master);
}


//====================================================================================
//
//====================================================================================

StringStream::MPX::MPX( const Multiplex::Factory* generator, Selector* master ):
         Multiplex( generator, master ),
         m_size(0)
{
}

StringStream::MPX::~MPX()
{
}


void StringStream::MPX::add( Selectable* resource, int mode )
{
   StringStream* ss = static_cast<StringStream*>(resource->instance());

   if( (mode & Selector::mode_write) != 0)
   {
      // always writeable
      onReadyWrite(resource);
   }

   if( (mode & Selector::mode_read) != 0)
   {
      ss->m_b->m_mtx.lock();
      uint32 bsize =  ss->m_b->m_str->size();
      if ( bsize > ss->m_posRead )
      {
         ss->m_b->m_mtx.unlock();
         onReadyRead( resource );
         return;
      }

      bool bNew = ss->m_b->m_waiters.insert( this ).second;
      ss->m_b->m_mtx.unlock();

      if( bNew )
      {
         if( ss->m_selectable != 0 )
         {
            ss->m_selectable->decref();
         }
         ss->m_selectable = resource;
         resource->incref();
         incref();
         m_size++;
      }
   }
}


void StringStream::MPX::remove( Selectable* resource )
{
   StringStream* ss = static_cast<StringStream*>(resource->instance());

   ss->m_b->m_mtx.unlock();
   bool bRemoved = ss->m_b->m_waiters.erase(this) > 0;
   ss->m_b->m_mtx.unlock();

   if( bRemoved )
   {
      if( ss->m_selectable != 0 )
      {
         ss->m_selectable->decref();
         ss->m_selectable = 0;
      }
      decref();
      m_size--;
   }
}

void StringStream::MPX::onStringStreamReady( StringStream* ss )
{
   fassert( ss->m_selectable != 0  );
   onReadyRead( ss->m_selectable );
   decref();
}

}

/* end of stringstream.cpp */
