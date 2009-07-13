/*
   FALCON - The Falcon Programming Language
   FILE: tokenizer.h

   Utility to parse complex and lengty strings.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 04 Feb 2009 16:19:28 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/tokenizer.h>
#include <falcon/rosstream.h>
#include <falcon/error.h>
#include <falcon/fassert.h>
#include <falcon/vm.h>

namespace Falcon {

Tokenizer::Tokenizer( TokenizerParams &params, const String &seps, Stream *ins, bool bOwn ):
   m_separators( seps ),
   m_params( params ),
   m_input(ins),
   m_bOwnStream( bOwn ),
   m_version(0)
{
   if ( seps == "" )
      m_separators = " ";
}

Tokenizer::Tokenizer( TokenizerParams &params, const String &seps, const String &source ):
   m_separators( seps ),
   m_params( params ),
   m_input( new ROStringStream( source ) ),
   m_bOwnStream( true ),
   m_version(0)
{
   if ( seps == "" )
      m_separators = " ";
}

Tokenizer::Tokenizer( const Tokenizer &other ):
   m_separators( other.m_separators ),
   m_params( other.m_params ),
   m_input( other.m_input != 0 ? dyncast<Stream*>(other.m_input->clone()) : 0 ),
   m_bOwnStream( true ),
   m_version(other.m_version)
{
}

Tokenizer::~Tokenizer()
{
   if ( m_bOwnStream )
      delete m_input;
}


CoreIterator *Tokenizer::getIterator( bool tail )
{
   // moved?
   if ( m_version != 0 )
      rewind();

   return new TokenIterator( this );
}


bool Tokenizer::next()
{
   // must be called when ready
   fassert( m_input != 0 );

   if ( m_input->eof() )
      return false;

   // reset the input buffer
   m_temp.size(0);
   m_version++;


   uint32 chr;
   while( m_input->get( chr ) )
   {
      bool skip = false;

      // is this character a separator?
      for( uint32 i = 0; i < m_separators.length(); i ++ )
      {
         if( chr == m_separators.getCharAt(i) )
         {
            // Yes. should we return now?
            if ( m_params.isGroupSep() && m_temp.size() == 0 )
            {
               skip = true;
               break; // we have to decide what to do outside.
            }

            // should we pack the thing?
            if( m_params.isBindSep() )
               m_temp+=chr;

            if ( m_params.isTrim() )
               m_temp.trim();

            return true;

         }
      }

      if ( skip )
         continue;

      // no, add this character
      m_temp += chr;

      if( m_params.maxToken() > 0 && m_temp.length() >= (uint32) m_params.maxToken() )
      {
         if ( m_params.isTrim() )
            m_temp.trim();

         return true;
      }
   }

   if ( m_params.isTrim() )
      m_temp.trim();
   // ok we can't find any more character; but is this a valid token?
   return m_temp.size() != 0 || ! m_params.isGroupSep();
}

bool Tokenizer::empty() const
{
   return m_input == 0 || m_input->eof();
}

void Tokenizer::rewind()
{
   if( m_input != 0 )
   {
      m_input->seekBegin(0);
      m_version = 0;
   }
}


FalconData* Tokenizer::clone() const
{
   return new Tokenizer( *this );
}


const Item &Tokenizer::front() const
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
      .origin( e_orig_runtime )
      .extra( "Tokenizer::front" ) );
}

const Item &Tokenizer::back() const
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ ).origin( e_orig_runtime ).extra( "Tokenizer::back" ) );
}

bool Tokenizer::insert( CoreIterator *iter, const Item &data )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ ).origin( e_orig_runtime ).extra( "Tokenizer::insert" ) );
}

bool Tokenizer::erase( CoreIterator *iter )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ ).origin( e_orig_runtime ).extra( "Tokenizer::erase" ) );
}

void Tokenizer::clear()
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ ).origin( e_orig_runtime ).extra( "Tokenizer::clear" ) );
}

void Tokenizer::append( const Item& itm )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ ).origin( e_orig_runtime ).extra( "Tokenizer::append" ) );
}

void Tokenizer::prepend( const Item& itm )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ ).origin( e_orig_runtime ).extra( "Tokenizer::prepend" ) );
}

void Tokenizer::parse( const String &data )
{
   if ( m_bOwnStream )
      delete m_input;

   m_input = new ROStringStream( data );
   m_bOwnStream = true;
   m_version++;
}


void Tokenizer::parse( Stream *in, bool bOwn )
{
   if ( m_bOwnStream )
      delete m_input;

   m_input = in;
   m_bOwnStream = bOwn;
   m_version++;
}


//=====================================================
// Iterator
//

TokenIterator::TokenIterator( Tokenizer *owner ):
   m_owner( owner ),
   m_version( owner->m_version )
{
}

TokenIterator::TokenIterator( const TokenIterator &other ):
   m_owner( other.m_owner ),
   m_version( other.m_owner == 0 ? 0 : other.m_owner->m_version )
{
}

bool TokenIterator::next()
{
   if( m_owner != 0 && m_version == m_owner->m_version )
   {
      bool v = m_owner->next();
      m_version = m_owner->m_version;
      return v;
   }
   return false;
}

bool TokenIterator::hasNext() const
{
   if( m_owner != 0 && m_version == m_owner->m_version )
   {
      return !m_owner->empty();
   }
   return false;

}

Item &TokenIterator::getCurrent() const
{
   fassert( m_owner != 0 && m_version == m_owner->m_version );

   // force to load the initial token.
   if ( m_version == 0 )
   {
      m_owner->next();
      m_version = m_owner->m_version;
   }

   CoreString* ret = new CoreString( m_owner->getToken() );
   ret->bufferize();

   m_cacheItem.setString( ret );
   return m_cacheItem;
}

bool TokenIterator::isOwner( void *collection ) const
{
   return m_owner == collection;
}

bool TokenIterator::equal( const CoreIterator &other ) const
{
   return other.isOwner( m_owner );
}

void TokenIterator::invalidate()
{
   m_owner = 0;
}

bool TokenIterator::prev()
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ ).origin( e_orig_runtime ).extra( "TokenIterator::prev" ) );
}

bool TokenIterator::hasPrev() const
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ ).origin( e_orig_runtime ).extra( "TokenIterator::hasPrev" ) );
}

bool TokenIterator::isValid() const
{
   return m_version == m_owner->m_version && m_owner->m_input != 0;
}

bool TokenIterator::erase()
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ ).origin( e_orig_runtime ).extra( "TokenIterator::erase" ) );
}

bool TokenIterator::insert( const Item &item )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ ).origin( e_orig_runtime ).extra( "TokenIterator::insert" ) );
}

FalconData* TokenIterator::clone() const
{
   return new TokenIterator( *this );
}


}
