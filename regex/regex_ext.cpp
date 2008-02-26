/*
   FALCON - The Falcon Programming Language.
   FILE: regex_ext.cpp
   $Id: regex_ext.cpp,v 1.14 2007/08/18 13:23:53 jonnymind Exp $

   Regular expression module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab mar 11 2006
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Regex
*/

#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/memory.h>
#include <falcon/fassert.h>
#include <falcon/autocstring.h>

#include <string.h>

#include "regex_ext.h"
#include "regex_mod.h"

namespace Falcon {
namespace Ext {

/**
   Initialization of the regex structure.
*/

FALCON_FUNC Regex_init( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *param = vm->param(0);
   Item *options = vm->param(1);

   if( param == 0 || ! param->isString() || ( options != 0 && ! options->isString() ) )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S, [S]" ) ) );
      return;
   }

   // read pattern options

   int optVal = 0;
   bool bStudy = false;

   if( options != 0 )
   {
      String *optString = options->asString();
      for( uint32 i = 0; i < optString->length(); i++ )
      {
         switch ( optString->getCharAt( i ) )
         {
            case 'a': optVal |= PCRE_ANCHORED; break;
            case 'i': optVal |= PCRE_CASELESS; break;
            case 'm': optVal |= PCRE_MULTILINE; break;
            case 's': optVal |= PCRE_DOTALL; break;
            case 'f': optVal |= PCRE_FIRSTLINE; break;
            case 'g': optVal |= PCRE_UNGREEDY; break;
            case 'S': bStudy = true; break;
            default:
               vm->raiseModError( new ParamError( ErrorParam( e_param_range, __LINE__ ).
                  extra( "Unrecognized option in pattern options" ) ) );
         }
      }
   }

   // determine the type of the string.
   String *source = param->asString();
   pcre *pattern;
   int errCode;
   const char *errDesc;
   int errOffset;

   if( source->manipulator()->charSize() == 1 )
   {
		char *stringData = (char *) memAlloc( source->size() + 1);
		memcpy( stringData, source->getRawStorage(), source->size() );
		stringData[source->size()] = '\0';
      pattern = pcre_compile2(
         stringData,
         optVal,
         &errCode,
         &errDesc,
         &errOffset,
         0 );
		 memFree( stringData );
   }
   else
   {
      char *stringData;
      uint32 size = source->size() * 4 + 1;
      stringData = (char *) memAlloc( size );
      source->toCString( stringData, size );

      pattern = pcre_compile2(
         stringData,
         optVal | PCRE_UTF8 | PCRE_NO_UTF8_CHECK,
         &errCode,
         &errDesc,
         &errOffset,
         0 );

      memFree( stringData );
   }

   if ( pattern == 0 )
   {
      vm->raiseModError( new RegexError( ErrorParam( 1150, __LINE__ ).
         desc( "Invalid regular expression" ).extra( errDesc ) ) );
      return;
   }

   RegexCarrier *data = new RegexCarrier( pattern );
   self->setUserData( data );

   if ( bStudy )
   {
      data->m_extra = pcre_study( pattern, 0, &errDesc );
      if ( data->m_extra == 0 && errDesc != 0 )
      {
         vm->raiseModError( new RegexError( ErrorParam( 1151, __LINE__ ).
            desc( "Error while studing the regular expression" ).extra( errDesc ) ) );
         return;
      }
   }
}

/**
   Regex.study()
   Will just take some extra time to study the data.
*/
FALCON_FUNC Regex_study( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   RegexCarrier *data = ( RegexCarrier *) self->getUserData();

   if( data->m_extra != 0 )
   {
      // already studied
      return;
   }

   const char *errDesc;
   data->m_extra = pcre_study( data->m_pattern, 0, &errDesc );
   if ( data->m_extra == 0 && errDesc != 0 )
   {
      vm->raiseModError( new RegexError( ErrorParam( 1151, __LINE__ ).
         desc( "Error while studing the regular expression" ).extra( errDesc ) ) );
   }
}


static int utf8_fwd_displacement( const char *stringData, int pos )
{
   int ret = 0;
   while( pos > 0  && stringData[ ret ] != 0 )
   {
      byte in = stringData[ ret ];

      // pattern 1111 0xxx
      if ( (in & 0xF8) == 0xF0 )
      {
         ret += 4;
      }
      // pattern 1110 xxxx
      else if ( (in & 0xF0) == 0xE0 )
      {
         ret += 3;
      }
      // pattern 110x xxxx
      else if ( (in & 0xE0) == 0xC0 )
      {
         ret += 2;
      }
      else if( in < 0x80 )
      {
         ret += 1;
      }
      // invalid pattern
      else {
         return -1;
      }

      pos--;
   }

   if ( pos == 0 )
      return ret;
   else
      return -1;
}



static int utf8_back_displacement( const char *stringData, int pos )
{
   int ret = 0;
   int pos1 = 0;
   while( pos1 < pos )
   {
      byte in = stringData[ pos1 ];

      // pattern 1111 0xxx
      if ( (in & 0xF8) == 0xF0 )
      {
         pos1 += 4;
      }
      // pattern 1110 xxxx
      else if ( (in & 0xF0) == 0xE0 )
      {
         pos1 += 3;
      }
      // pattern 110x xxxx
      else if ( (in & 0xE0) == 0xC0 )
      {
         pos1 += 2;
      }
      else if( in < 0x80 )
      {
         pos1 += 1;
      }
      // invalid pattern
      else {
         return -1;
      }

      ret ++;
   }

   return ret;
}


static void internal_regex_match( RegexCarrier *data, String *source, int from )
{

   if( source->manipulator()->charSize() == 1 )
   {
      data->m_matches = pcre_exec(
         data->m_pattern,
         data->m_extra,
         (const char *) source->getRawStorage(),
         source->size(),
         from,
         0,
         data->m_ovector,
         data->m_ovectorSize );
   }
   else
   {
      AutoCString stringData( *source );

      // displace the from indicator to match utf-8
      from = utf8_fwd_displacement( stringData.c_str(), from );
      if ( from == -1 )
      {
         data->m_matches = PCRE_ERROR_BADUTF8;
         return;
      }

      data->m_matches = pcre_exec(
         data->m_pattern,
         data->m_extra,
         (char *) stringData.c_str(),
         stringData.length(),
         from,
         PCRE_NO_UTF8_CHECK,
         data->m_ovector,
         data->m_ovectorSize );

      for( int i = 0; i < data->m_matches; i++ )
      {
         data->m_ovector[ i * 2 ] = utf8_back_displacement( stringData, data->m_ovector[ i * 2 ] );
         data->m_ovector[ i * 2 + 1 ] = utf8_back_displacement( stringData, data->m_ovector[ i * 2 + 1] );
      }
   }
}

/**
   Regex.match( string ) --> bMatch

   Returns true if the regular expression matches the given string.
*/
FALCON_FUNC Regex_match( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   RegexCarrier *data = ( RegexCarrier *) self->getUserData();

   Item *source = vm->param(0);

   if( source == 0 || ! source->isString() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S" ) ) );
      return;
   }

   internal_regex_match( data, source->asString(), 0 );

   if ( data->m_matches == PCRE_ERROR_NOMATCH )
   {
      vm->retval( (int64) 0 );
      return;
   }

   if ( data->m_matches < 0 )
   {
      String errVal = "Internal error: ";
      errVal.writeNumber( (int64) data->m_matches );
      vm->raiseModError( new RegexError( ErrorParam( 1152, __LINE__ ).
         desc( "Error while matching the regular expression" ).extra( errVal ) ) );
      return;
   }

   vm->retval( (int64) 1 );
}

/**
   Regex.find( string [, start] ) --> [range]

   Finds the first occourence of the pattern in the string. The returned ranged
   may be applied to the string in order to extract the desired substring.

   If the pattern doesn't matches, returns nil.
*/
FALCON_FUNC Regex_find( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   RegexCarrier *data = ( RegexCarrier *) self->getUserData();

   Item *source = vm->param(0);
   Item *from_i = vm->param(1);

   if( source == 0 || ! source->isString() || ( from_i != 0 && ! from_i->isOrdinal() ) )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S, [I]" ) ) );
      return;
   }

   int from = 0;
   if ( from_i != 0 )
   {
      from = (int) from_i->forceInteger();
      if ( from < 0 )
         from = 0;
   }

   internal_regex_match( data, source->asString(), from );

   if ( data->m_matches >= 0  )
   {
      // we know by hypotesis that oVector is at least 1 entry.
      Item rng;
      rng.setRange( data->m_ovector[0], data->m_ovector[1], false );
      vm->retval( rng );
   }
   else if ( data->m_matches == PCRE_ERROR_NOMATCH ){
      vm->retnil();
   }
   else
   {
      String errVal = "Internal error: ";
      errVal.writeNumber( (int64) data->m_matches );
      vm->raiseModError( new RegexError( ErrorParam( 1152, __LINE__ ).
         desc( "Error while matching the regular expression" ).extra( errVal ) ) );
   }
}

static void internal_findAll( Falcon::VMachine *vm, bool overlapped )
{
   CoreObject *self = vm->self().asObject();
   RegexCarrier *data = ( RegexCarrier *) self->getUserData();
   Item *source = vm->param(0);
   Item *from_i = vm->param(1);
   Item *maxCount_i = vm->param(2);

   if( source == 0 || ! source->isString() ||
      ( from_i != 0 && ! from_i->isOrdinal() ) ||
      ( maxCount_i != 0 && ! maxCount_i->isOrdinal() )
      )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S, [I], [I]" ) ) );
      return;
   }

   int from = 0;
   if ( from_i != 0 )
   {
      from = (int) from_i->forceInteger();
      if ( from < 0 )
         from = 0;
   }

   uint32 max = 0xFFFFFFFF;
   if ( maxCount_i != 0 )
   {
      max = (int) maxCount_i->forceInteger();
      if ( max <= 0 )
         max = 0xFFFFFFFF;
   }

   CoreArray *ca = new CoreArray( vm );
   int frontOrBack = overlapped ? 0 : 1;
   uint32 maxLen = source->asString()->length();

   do {
      internal_regex_match( data, source->asString(), from );
      if( data->m_matches > 0 )
      {
         Item rng;
         rng.setRange( data->m_ovector[0], data->m_ovector[1], false );
         ca->append( rng );
         // restart from the end of the patter
         from = data->m_ovector[frontOrBack];
         // as we're going to exit.
      }
      max--;
   } while( data->m_matches > 0 && max > 0 && from < (int32) maxLen );

   if ( data->m_matches < 0 && data->m_matches != PCRE_ERROR_NOMATCH )
   {
      String errVal = "Internal error: ";
      errVal.writeNumber( (int64) data->m_matches );
      vm->raiseModError( new RegexError( ErrorParam( 1152, __LINE__ ).
         desc( "Error while matching the regular expression" ).extra( errVal ) ) );
      return;
   }

   // always return an array, even if empty
   vm->retval( ca );
}

/**
   Regex.findAll( string [, start] [,maxcount] ) --> [ vector of ranges ]

   Finds the all the occourences of the pattern in the string.
   Doesn't check for overlapped matches

   After findall, captured expressions can't be searched

   If the pattern doesn't matches, returns nil.
*/
FALCON_FUNC Regex_findAll( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   RegexCarrier *data = ( RegexCarrier *) self->getUserData();

   internal_findAll( vm, false );
}

/**
   Regex.findAllOverlapped( string [, start] [,maxcount] ) --> [ vector of ranges ]

   Finds the all the occourences of the pattern in the string.
   Doesn't check for overlapped matches

   After findall, captured expressions can't be retreived.

   This versions scans also for overlapped pattenrs.

   If the pattern doesn't matches, returns nil.
*/
FALCON_FUNC Regex_findAllOverlapped( ::Falcon::VMachine *vm )
{
   internal_findAll( vm, true );
}


/**
   Regex.replace( strfrom, strRep, [start] ) --> replacedString

   Returns the replaced string.
*/
FALCON_FUNC Regex_replace( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   RegexCarrier *data = ( RegexCarrier *) self->getUserData();

   Item *source_i = vm->param(0);
   Item *dest_i = vm->param(1);
   Item *from_i = vm->param(2);

   if( source_i == 0 || ! source_i->isString() || dest_i == 0 || ! dest_i->isString() ||
      ( from_i != 0 && ! from_i->isOrdinal() )
      )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S, S, [I]" ) ) );
      return;
   }

   int from = 0;
   if ( from_i != 0  )
   {
      from = (int) from_i->forceInteger();
      if ( from < 0 )
         from = 0;
   }

   String *source = source_i->asString();
   String *dest = dest_i->asString();

   internal_regex_match( data, source, from );

   if ( data->m_matches == PCRE_ERROR_NOMATCH )
   {
      vm->retval( source );
      return;
   }

   if ( data->m_matches < 0 )
   {
      String errVal = "Internal error: ";
      errVal.writeNumber( (int64) data->m_matches );
      vm->raiseModError( new RegexError( ErrorParam( 1152, __LINE__ ).
         desc( "Error while matching the regular expression" ).extra( errVal ) ) );
      return;
   }

   source->change( data->m_ovector[0], data->m_ovector[1], *dest );
   vm->retval( source );
}


/**
   Regex.replaceAll( strfrom, strRep, [maxCount] ) --> bMatch

   Returns true if the regular expression matches the given string.
*/
FALCON_FUNC Regex_replaceAll( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   RegexCarrier *data = ( RegexCarrier *) self->getUserData();

   Item *source_i = vm->param(0);
   Item *dest_i = vm->param(1);
   Item *max_i = vm->param(2);

   if( source_i == 0 || ! source_i->isString() || dest_i == 0 || ! dest_i->isString() ||
      ( max_i != 0 && ! max_i->isOrdinal() )
   )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S, S, [I]" ) ) );
      return;
   }

   uint32 max = 0xFFFFFFFF;
   if ( max_i != 0 )
   {
      max = (int) max_i->forceInteger();
      if ( max <= 1 )
         max = 0xFFFFFFFF;
   }

   String *source = source_i->asString();
   String *clone = 0;
   String *dest = dest_i->asString();
   uint32 destLen = dest->length();

   int from = 0;
   do {
      internal_regex_match( data, source, from );
      if( data->m_matches > 0 )
      {
         if ( clone == 0 ) {
            clone = new GarbageString( vm, *source );
            source = clone;
         }
         source->change( data->m_ovector[0], data->m_ovector[1], *dest );
         from = data->m_ovector[0] + destLen;
         // as we're going to exit.
      }
      max--;
   } while( data->m_matches > 0 && max > 0 && from < (int32) source->length() );


   if ( data->m_matches < 0 && data->m_matches != PCRE_ERROR_NOMATCH )
   {
      String errVal = "Internal error: ";
      errVal.writeNumber( (int64) data->m_matches );
      vm->raiseModError( new RegexError( ErrorParam( 1152, __LINE__ ).
         desc( "Error while matching the regular expression" ).extra( errVal ) ) );
      return;
   }

   if ( clone != 0 )
      vm->retval( clone );
   else
      vm->retnil();
}


/**
   Returns the count of captured entries this Regex returns on a successful match.
*/
FALCON_FUNC Regex_capturedCount( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   RegexCarrier *data = ( RegexCarrier *) self->getUserData();

   if( data->m_matches > 0 )
      vm->retval( (int64) data->m_matches );
   else
      vm->retval( (int64) 0 );
}
/**
   Returns a range describing the nth captured pattern.
   captured( 0 ) returns a range describing the WHOLE captured pattern, so
   a loop retreiving all the captured patterns should be from 0 to self.capturedCount()
   INCLUDED.

   Raises an error if the entry has not been assigned.
*/
FALCON_FUNC Regex_captured( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   RegexCarrier *data = ( RegexCarrier *) self->getUserData();

   Item *pos_i = vm->param(0);
   if( pos_i == 0 || ! pos_i->isOrdinal() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "I" ) ) );
      return;
   }


   // valid also if we didn't get a good count
   int maxCount = data->m_matches;
   int count = (int) pos_i->forceInteger();

   if ( count < 0 ||  count >= maxCount )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_param_range, __LINE__ ).
         extra( "Captured ID out of range." ) ) );
      return;
   }

   Item rng;
   rng.setRange( data->m_ovector[ count * 2 ] , data->m_ovector[ count * 2 + 1 ], false );
   vm->retval( rng );
}



FALCON_FUNC Regex_grab( Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   RegexCarrier *data = ( RegexCarrier *) self->getUserData();
   Item *source = vm->param(0);

   if( source == 0 || ! source->isString() )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "S" ) ) );
      return;
   }

   internal_regex_match( data, source->asString(), 0 );

   if ( data->m_matches == PCRE_ERROR_NOMATCH )
   {
      vm->retnil();
      return;
   }

   if ( data->m_matches < 0 )
   {
      String errVal = "Internal error: ";
      errVal.writeNumber( (int64) data->m_matches );
      vm->raiseModError( new RegexError( ErrorParam( 1152, __LINE__ ).
         desc( "Error while matching the regular expression" ).extra( errVal ) ) );
      return;
   }

   // grab all the strings

   CoreArray *ca = new CoreArray( vm );
   for( int32 capt = 0; capt < data->m_matches; capt++ )
   {

      String *grabbed = new GarbageString( vm,
            source->asString()->subString(
               data->m_ovector[ capt * 2 ], data->m_ovector[ capt * 2 + 1 ] )
               );
      ca->append( grabbed );
   }

   vm->retval( ca );
}


FALCON_FUNC Regex_compare( Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   RegexCarrier *data = ( RegexCarrier *) self->getUserData();
   Item *source = vm->param(0);

   if( source == 0 )
   {
      vm->raiseModError( new  ParamError( ErrorParam( e_inv_params, __LINE__ ).
         extra( "X" ) ) );
      return;
   }

   // minimal match vector
   int ovector[3];

   // If the source is a string, perform a non-recorded match
   if ( source->isString() )
   {
      bool match;
      String *str = source->asString();

      if( str->manipulator()->charSize() == 1 )
      {
         match = 0 < pcre_exec(
            data->m_pattern,
            data->m_extra,
            (const char *) str->getRawStorage(),
            str->size(),
            0,
            0,
            ovector,
            3 );
      }
      else
      {
         AutoCString src( *str );

         match = 0 < pcre_exec(
            data->m_pattern,
            data->m_extra,
            src.c_str(),
            src.length(),
            0,
            PCRE_NO_UTF8_CHECK,
            ovector,
            3 );
      }

      if ( match )
         vm->retval( (int64) 0 ); // zero means compare ==
      else
         vm->retnil(); // we can't decide. Let the VM do that for us.
   }
   // Otherwise return nil, that will tell VM to use default matching algos
   else {
      vm->retnil();
   }
}

FALCON_FUNC Regex_version( Falcon::VMachine *vm )
{
   const char *ver = pcre_version();
   vm->retval( new GarbageString( vm, ver, -1 ) );
}

FALCON_FUNC  RegexError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new RegexError ) );

   ::Falcon::core::Error_init( vm );
}

}
}

/* end of regex_ext.cpp */
