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
#include <falcon/eng_messages.h>
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


//============================================================
// Iterator management.
//============================================================

void Tokenizer::getIterator( Iterator& tgt, bool tail ) const
{
   // nothing to do
}


void Tokenizer::copyIterator( Iterator& tgt, const Iterator& source ) const
{
   // nothing to do
}


void Tokenizer::disposeIterator( Iterator& tgt ) const
{
   // no need to do anything
}


void Tokenizer::gcMarkIterator( Iterator& tgt ) const
{
   // no need to do anything
}

void Tokenizer::insert( Iterator &iter, const Item &data )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "Tokenizer::insert" ) );
}

void Tokenizer::erase( Iterator &iter )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "Tokenizer::erase" ) );
}


bool Tokenizer::hasNext( const Iterator &iter ) const
{
   return isReady();
}


bool Tokenizer::hasPrev( const Iterator &iter ) const
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "Tokenizer::hasPrev" ) );
}

bool Tokenizer::hasCurrent( const Iterator &iter ) const
{
   return isReady();
}


bool Tokenizer::next( Iterator &iter ) const
{
   return const_cast<Tokenizer*>(this)->next();
}


bool Tokenizer::prev( Iterator &iter ) const
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
            .origin( e_orig_runtime ).extra( "Tokenizer::prev" ) );
}

Item& Tokenizer::getCurrent( const Iterator &iter )
{
   static Item i_temp;
   i_temp = new CoreString( m_temp );
   return i_temp;
}


Item& Tokenizer::getCurrentKey( const Iterator &iter )
{
   throw new CodeError( ErrorParam( e_non_dict_seq, __LINE__ ) );
}


bool Tokenizer::equalIterator( const Iterator &first, const Iterator &second ) const
{
   return false;
}

bool Tokenizer::isValid( const Iterator &first ) const
{
   return true;
}

}
