/*
   FALCON - The Falcon Programming Language.
   FILE: string.cpp
   $Id: string.cpp,v 1.12 2007/08/11 10:22:36 jonnymind Exp $

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven nov 5 2004
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
   Short description
*/


#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/memory.h>
#include <falcon/membuf.h>

#include <string.h>

namespace Falcon {
namespace Ext {

FALCON_FUNC  strSplitTrimmed ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *target = vm->param(0);
   Item *splitstr = vm->param(1);
   Item *count = vm->param(2);
   uint32 limit;

   if ( target == 0 || ! target->isString() || splitstr == 0 || ! splitstr->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( count != 0 && ! count->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   limit = count == 0 ? 0xffffffff: (int32) count->forceInteger();

   // Parameter estraction.
   String *tg_str = target->asString();
   uint32 tg_len = target->asString()->length();

   String *sp_str = splitstr->asString();
   uint32 sp_len = splitstr->asString()->length();

   // Is the split string empty
   if ( sp_len == 0 ) {
      vm->retnil();
      return;
   }

   // return item.
   CoreArray *retarr = new CoreArray( vm );

   // if the token is wider than the string, just return the string
   if ( tg_len <= sp_len )
   {
	   retarr->append( new GarbageString( vm, *tg_str ) );
	   vm->retval( retarr );
	   return;
   }

   uint32 pos = 0;
   uint32 last_pos = 0;
   // scan the string
   while( limit > 1 && pos <= tg_len - sp_len  )
   {
      uint32 sp_pos = 0;
      // skip matching pattern-
      while ( tg_str->getCharAt( pos ) == sp_str->getCharAt( sp_pos ) &&
              sp_pos < sp_len && pos <= tg_len - sp_len ) {
         sp_pos++;
         pos++;
      }

      // a match?
      if ( sp_pos == sp_len ) {
         // put the item in the array.
         uint32 splitend = pos - sp_len;
         retarr->append( new GarbageString( vm, String( *tg_str, last_pos, splitend ) ) );
         last_pos = pos;
         limit--;
         // skip matching pattern
         while( sp_pos == sp_len && pos <= tg_len - sp_len ) {
            sp_pos = 0;
            last_pos = pos;
            while ( tg_str->getCharAt( pos ) == sp_str->getCharAt( sp_pos )
                    && sp_pos < sp_len && pos <= tg_len - sp_len ) {
               sp_pos++;
               pos++;
            }
         }
         pos = last_pos;

      }
      else
         pos++;
   }

   // Residual element?
   if ( limit >= 1 || last_pos < tg_len ) {
      uint32 splitend = tg_len;
      retarr->append( new GarbageString( vm, String( *tg_str, last_pos, splitend ) ) );
   }

   vm->retval( retarr );
}

FALCON_FUNC  strSplit ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *target = vm->param(0);
   Item *splitstr = vm->param(1);
   Item *count = vm->param(2);
   uint32 limit;

   if ( target == 0 || ! target->isString() || splitstr == 0 || ! splitstr->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( count != 0 && ! count->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   limit = count == 0 ? 0xffffffff: (int32) count->forceInteger();

   // Parameter estraction.
   String *tg_str = target->asString();
   uint32 tg_len = target->asString()->length();

   String *sp_str = splitstr->asString();
   uint32 sp_len = splitstr->asString()->length();

   // Is the split string empty
   if ( sp_len == 0 ) {
      vm->retnil();
      return;
   }

   // return item.
   CoreArray *retarr = new CoreArray( vm );

   // if the token is wider than the string, just return the string
   if ( tg_len <= sp_len )
   {
	   retarr->append( new GarbageString( vm, *tg_str ) );
	   vm->retval( retarr );
	   return;
   }

   uint32 pos = 0;
   uint32 last_pos = 0;
   // scan the string
   while( limit > 1 && pos <= tg_len - sp_len  )
   {
      uint32 sp_pos = 0;
      // skip matching pattern-
      while ( tg_str->getCharAt( pos ) == sp_str->getCharAt( sp_pos ) &&
              sp_pos < sp_len && pos <= tg_len - sp_len ) {
         sp_pos++;
         pos++;
      }

      // a match?
      if ( sp_pos == sp_len ) {
         // put the item in the array.
         uint32 splitend = pos - sp_len;
         retarr->append( new GarbageString( vm, String( *tg_str, last_pos, splitend ) ) );
         last_pos = pos;
         limit--;

      }
      else
         pos++;
   }

   // Residual element?
   if ( limit >= 1 || last_pos < tg_len ) {
      uint32 splitend = tg_len;
      retarr->append( new GarbageString( vm, String( *tg_str, last_pos, splitend ) ) );
   }

   vm->retval( retarr );
}


FALCON_FUNC  strMerge ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *source = vm->param(0);
   Item *mergestr = vm->param(1);
   Item *count = vm->param(2);
   uint64 limit;

   if ( source == 0 || ! source->isArray() || ( mergestr != 0 && ! mergestr->isString() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( count != 0 && ! count->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   // Parameter estraction.

   limit = count == 0 ? 0xffffffff: count->forceInteger();

   String *mr_str;
   if( mergestr != 0 )
   {
      mr_str = mergestr->asString();
   }
   else
      mr_str = 0;

   Item *elements = source->asArray()->elements();
   uint32 len = source->asArray()->length();
   if ( limit < len )
      len = (uint32) limit;

   String *ts = new GarbageString( vm );

   // filling the target.
   for( uint32 i = 0; i < len ; i ++ ) {
      if ( elements[i].type() != FLC_ITEM_STRING ) {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
         return;
      }
      String *src = elements[i].asString();
      ts->append( *src );
      if ( mr_str != 0 && i < len - 1 )
         ts->append( *mr_str );
   }

   vm->retval( ts );
}


FALCON_FUNC  strFind ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *target = vm->param(0);
   Item *needle = vm->param(1);
   Item *start_item = vm->param(2);
   Item *end_item = vm->param(3);

   if ( target == 0 || ! target->isString() || needle == 0 || ! needle->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( start_item != 0 && ! start_item->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( end_item != 0 && ! end_item->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int32 start = start_item == 0 ? 0 : (int32) start_item->forceInteger();
   int32 end = end_item == 0 ? csh::npos : (int32) end_item->forceInteger();

   uint32 pos = target->asString()->find( *needle->asString(), start, end );
   if ( pos != csh::npos )
      vm->retval( (int)pos );
   else
      vm->retval( -1 );
}

FALCON_FUNC  strBackFind ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *target = vm->param(0);
   Item *needle = vm->param(1);
   Item *start_item = vm->param(2);
   Item *end_item = vm->param(3);

   if ( target == 0 || ! target->isString() || needle == 0 || ! needle->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( start_item != 0 && ! start_item->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( end_item != 0 && ! end_item->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int32 start = start_item == 0 ? 0 : (int32) start_item->forceInteger();
   int32 end = end_item == 0 ? csh::npos : (int32) end_item->forceInteger();
   uint32 pos = target->asString()->rfind( *needle->asString(), start, end );
   if ( pos != csh::npos )
      vm->retval( (int)pos );
   else
      vm->retval( -1 );
}


FALCON_FUNC  strFront ( ::Falcon::VMachine *vm )
{
   Item *target = vm->param(0);
   Item *length = vm->param(1);

   if ( target == 0 || ! target->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( length == 0 || (length->type() != FLC_ITEM_INT && length->type() != FLC_ITEM_NUM )  ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int32 len = (int32) length->forceInteger();
   if ( len <= 0 ) {
      vm->retval( new GarbageString( vm,"") );
   }
   else if ( len > (int32) target->asString()->length() ) {
      vm->retval( new GarbageString( vm, *target->asString() ) );
   }
   else {
      vm->retval( target->asString()->subString(0, len ) );
   }
}

FALCON_FUNC  strBack ( ::Falcon::VMachine *vm )
{
   Item *target = vm->param(0);
   Item *length = vm->param(1);

   if ( target == 0 || ! target->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( length == 0 || (length->type() != FLC_ITEM_INT && length->type() != FLC_ITEM_NUM )  ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int32 len = (int32) length->forceInteger();
   int32 len1 = target->asString()->length();
   if ( len <= 0 ) {
      vm->retval( new GarbageString( vm,"") );
   }
   else if ( len >= len1 ) {
      vm->retval( new GarbageString( vm, *target->asString() ) );
   }
   else {
      vm->retval( target->asString()->subString( len1 - len ) );
   }
}

FALCON_FUNC  strTrim ( ::Falcon::VMachine *vm )
{
   Item *target = vm->param(0);

   if ( target == 0 || ! target->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   String *cs = new GarbageString( vm, *target->asString() );

   Item *trimChars = vm->param(1);
   if ( trimChars == 0 ) {
      cs->backTrim();
      vm->retval( cs );
   }
   else if ( ! trimChars->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }
   else {
      int32 pos = cs->length()-1;
      String *trim = trimChars->asString();
      int32 tLen = trim->length();

      while ( pos >= 0 ) {
         uint32 chr = cs->getCharAt( pos );
         int found = 0;

         for ( int32 tIdx=0; tIdx < tLen; tIdx++ )
            if ( chr == trim->getCharAt( tIdx ) )
               found = 1;
         if ( found == 0 )
            break;
         pos--;
      }
      // has something to be trimmed?
      if ( pos >= 0)
         vm->retval( cs->subString( 0, pos + 1 ) );
      else
         vm->retval( new GarbageString( vm ) );
   }
}

FALCON_FUNC  strFrontTrim ( ::Falcon::VMachine *vm )
{
   Item *target = vm->param(0);

   if ( target == 0 || ! target->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }


   String *cs = new GarbageString( vm, *target->asString() );

   Item *trimChars = vm->param(1);
   if (trimChars == 0 ) {
      cs->frontTrim();
      vm->retval( cs );
   }
   else if ( ! trimChars->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }
   else {
      int pos = 0;
      int32 len = cs->length();
      String *trim = trimChars->asString();
      int32 tLen = trim->length();

      while( pos <= len )
      {
         uint32 chr = cs->getCharAt( pos );
         int found = 0;

         for ( int32 tIdx = 0; tIdx < tLen; tIdx++ )
            if ( chr == trim->getCharAt( tIdx ) )
               found = 1;
         if ( found == 0 )
            break;
         pos++;
      }

      // has something to be trimmed?
      if ( pos < len )
         vm->retval( cs->subString( pos, len ) );
      else
         vm->retval( new GarbageString( vm ) );
   }
}

FALCON_FUNC  strAllTrim ( ::Falcon::VMachine *vm )
{
   Item *target = vm->param(0);

   if ( target == 0 || ! target->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   String *cs = new GarbageString( vm, *target->asString() );

   Item *trimChars = vm->param(1);
   if ( trimChars == 0 ) {
      cs->trim();
      vm->retval( cs );
   }
   else if ( ! trimChars->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
   }
   else {
      String *trim = trimChars->asString();
      int32 tLen = trim->length();
      int32 len = cs->length();
      int32 start = 0;
      int32 end = len;
      uint32 chr;
      int found = 0;

      while( start < len )
      {
         found = 0;
         chr = cs->getCharAt( start );
         for ( int32 tIdx=0; tIdx < tLen; tIdx++ )
            if ( chr == trim->getCharAt( tIdx ) )
               found = 1;
         if ( found == 0 )
            break;
         start++;
      }

      while( end > start )
      {
         found = 0;
         chr = cs->getCharAt( end - 1 );
         for ( int32 tIdx=0; tIdx < tLen; tIdx++ )
            if ( chr == trim->getCharAt( tIdx ) )
               found = 1;
         if ( found == 0 )
            break;
         end--;
      }

      // an empty string if set is empty
      vm->retval( cs->subString( start, end ) );
   }

}

FALCON_FUNC  strReplace ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *target = vm->param(0);
   Item *needle = vm->param(1);
   Item *replacer = vm->param(2);
   Item *start_item = vm->param(3);
   Item *end_item = vm->param(4);

   if ( target == 0 || ! target->isString() || needle == 0 || ! needle->isString() ||
         replacer == 0 || ! replacer->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( start_item != 0 && ! start_item->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( end_item != 0 && ! end_item->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   // Parameter estraction.
   String *tg_str = target->asString();
   uint32 tg_len = target->asString()->length();

   String *ned_str = needle->asString();
   int32 ned_len = (int32) needle->asString()->length();

   // Is the needle is empty
   if ( ned_len == 0 ) {
      // shallow copy the target
      vm->retval( *target );
      return;
   }

   String *rep_str = replacer->asString();
   uint32 rep_len = replacer->asString()->length();

   int32 start = start_item ? (int32) start_item->forceInteger(): 0;
   if ( start < 0 ) start = 0;
   int32 end = end_item ? (int32) end_item->forceInteger(): tg_len-1;
   if ( end >= (int32) tg_len ) end = tg_len-1;

   String *ret = new GarbageString( vm );
   if ( start > 0 )
      ret->append( String( *tg_str, 0, start ) );
   int32 old_start = start;
   while ( start <= end )
   {
      int32 ned_pos = 0;
      int32 pos = 0;
      // skip matching pattern
      while ( tg_str->getCharAt( start + pos ) == ned_str->getCharAt( ned_pos )
               && ned_pos < (int32) ned_len && start + ned_pos <= end )
      {
         ned_pos++;
         pos++;
      }

      // a match?
      if ( ned_pos == ned_len )
      {
         if ( start > old_start ) {
            ret->append( String( *tg_str, old_start, start ) );
         }

         if ( rep_len > 0 ) {
            ret->append( *rep_str );
         }

         start += ned_len;
         old_start = start;
      }
      else
         start++;
   }

   if ( old_start < (int32)tg_len )
      ret->append( String( *tg_str, old_start ) );

   vm->retval( ret );
}

FALCON_FUNC  strReplicate ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *strrep = vm->param(0);
   Item *qty = vm->param(1);

   if ( strrep == 0 || strrep->type() != FLC_ITEM_STRING || qty == 0 || ! qty->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int32 repl = (int32) qty->forceInteger();
   String *replicated = strrep->asString();
   int32 len = replicated->size() * repl;
   if ( len <= 0 ) {
      vm->retval( new GarbageString( vm,"") );
      return;
   }

   String *target = new GarbageString( vm );
   target->reserve( len );

   int pos = 0;
   while ( pos < len ) {
      memcpy( target->getRawStorage() + pos, replicated->getRawStorage(), replicated->size() );
      pos+= replicated->size();
   }
   target->manipulator( const_cast<Falcon::csh::Base*>(replicated->manipulator()->bufferedManipulator()) );
   target->size( len );
   vm->retval( target );
}

FALCON_FUNC  strBuffer ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *qty = vm->param(0);
   if ( qty == 0 || ! qty->isOrdinal() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   int32 size = (int32) qty->forceInteger();
   if ( size <= 0 ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }


   vm->retval( new GarbageString( vm, String( size ) ) );
}

FALCON_FUNC  strUpper ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *source = vm->param(0);
   if ( source == 0 || ! source->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   String *src = source->asString();
   if ( src->size() == 0 )
   {
      vm->retval( new GarbageString( vm ) );
   }
   else {
      String *target = new GarbageString( vm, *src );
      target->upper();
      vm->retval( target );
   }
}

FALCON_FUNC  strLower ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *source = vm->param(0);
   if ( source == 0 || ! source->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   String *src = source->asString();
   if ( src->size() == 0 )
   {
      vm->retval( new GarbageString( vm ) );
   }
   else {
      String *target = new GarbageString( vm, *src );
      target->lower();
      vm->retval( target );
   }
}

FALCON_FUNC  strCmpIgnoreCase ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *s1_itm = vm->param(0);
   Item *s2_itm = vm->param(1);
   if ( s1_itm == 0 || ! s1_itm->isString() || s2_itm == 0 || !s2_itm->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   String *str1 = s1_itm->asString();
   String *str2 = s2_itm->asString();

   int32 len1 = str1->length();
   int32 len2 = str2->length();

   int32 minlen = len1 > len2 ? len2 : len1;

   for( int32 i = 0; i < minlen; i ++ )
   {
      int32 elem1 = str1->getCharAt( i );
      int32 elem2 = str2->getCharAt( i );
      if ( elem1 >= 'A' && elem1 <= 'Z' )
         elem1 |= 0x20;
      if ( elem2 >= 'A' && elem2 <= 'Z' )
         elem2 |= 0x20;

      if ( elem1 > elem2 ) {
         vm->retval( 1 );
         return;
      }

      if ( elem1 < elem2 ) {
         vm->retval( -1 );
         return;
      }
   }

   if ( len1 > len2 ) {
      vm->retval( 1 );
      return;
   }

   if ( len1 < len2 ) {
      vm->retval( -1 );
      return;
   }

   // same!!!
   vm->retval( 0 );
}


FALCON_FUNC  strWildcardMatch ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *s1_itm = vm->param(0);
   Item *s2_itm = vm->param(1);
   Item *i_bIcase = vm->param(2);
   if ( s1_itm == 0 || ! s1_itm->isString() || s2_itm == 0 || !s2_itm->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
           extra("S,S") ) );
      return;
   }

   // Ignore case?
   bool bIcase = i_bIcase == 0 ? false : i_bIcase->isTrue();

   // The first is the wildcard, the second is the matched thing.
   String *wcard = s1_itm->asString();
   String *cfr = s2_itm->asString();
   uint32 wpos = 0, wlen = wcard->length();
   uint32 cpos = 0, clen = cfr->length();

   uint32 wstarpos = 0xFFFFFFFF;


   while ( cpos < clen )
   {
      if( wpos == wlen )
      {
         // we have failed the match; but if we had a star, we
         // may roll back to the starpos and try to match the
         // rest of the string
         if ( wstarpos != 0xFFFFFFFF )
         {
            wpos = wstarpos;
         }
         else {
            // no way, we're doomed.
            break;
         }
      }

      uint32 wchr = wcard->getCharAt( wpos );
      uint32 cchr = cfr->getCharAt( cpos );

      switch( wchr )
      {
         case '?': // match any character
            wpos++;
            cpos++;
         break;

         case '*':
         {
            // mark for restart in case of bad match.
            wstarpos = wpos;

            // match till the next character
            wpos++;
            // eat all * in a row
            while( wpos < wlen )
            {
               wchr = wcard->getCharAt( wpos );
               if ( wchr != '*' )
                  break;
               wpos++;
            }

            if ( wpos == wlen )
            {
               // we have consumed all the chars
               cpos = clen;
               break;
            }


            //eat up to next character
            while( cpos < clen )
            {
               cchr = cfr->getCharAt( cpos );
               if ( cchr == wchr )
                  break;
               cpos ++;
            }

            // we have eaten up the same char? --  then advance also wpos to prepare next loop
            if ( cchr == wchr )
            {
               wpos++;
               cpos++;
            }
            // else, everything must stay as it is, so cpos == clen but wpos != wlen causing fail.
         }
         break;

         default:
            if ( cchr == wchr ||
                  ( bIcase && cchr < 128 && wchr < 128 && (cchr | 32) == (wchr | 32) )
               )
            {
               cpos++;
               wpos++;
            }
            else
            {
               // can we retry?
               if ( wstarpos != 0xFFFFFFFF )
                  wpos = wstarpos;
               else {
                  // check failed -- we're doomed
                  vm->retval( false );
                  return;
               }
            }
      }
   }

   // at the end of the loop, the match is ok only if both the cpos and wpos are at the end
   vm->retval( wpos == wlen && cpos == clen );
}

/*# function strToMemBuf
   @param string String to be converted in a membuf
   @optparam wordWidth The memory buffer word width (defaults to string character size)
   @brief Convets a string to a membuf
   @return The resulting membuf
   
   This function creates a membuf from a string. The resulting membuf
   has the same word width of the original string, which may be 1, 2 or 4
   byte wide depending on the size needed to store its contents. It is possible
   to specify a different word width; in that case the function will be much
   less efficient (each character must be copied).

   If wordWidth is set to zero, the resulting memory buffer will have 1 byte
   long elements, but the content of the string will be copied as-is, bytewise,
   regardless of its character size.
*/

FALCON_FUNC  strToMemBuf ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *i_string = vm->param(0);
   Item *i_wordWidth = vm->param(1);

   if( i_string == 0 || ! i_string->isString() ||
      ( i_wordWidth != 0 && ! i_wordWidth->isOrdinal() )
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
           extra("S,[N]") ) );
      return;
   }

   String *string = i_string->asString();
   int charSize = string->manipulator()->charSize();
   int64 ww = i_wordWidth == 0 ? charSize : i_wordWidth->forceInteger();
   MemBuf *result;

   if ( ww == 0 )
   {
      result = new MemBuf_1( vm, string->size() );
      memcpy( result->data(), string->getRawStorage(), string->size() );
   }
   else 
   {
      result = MemBuf::create( vm, charSize, string->length() );

      if ( result == 0 )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_param_range, __LINE__ ).origin( e_orig_runtime ).
           extra("0-4") ) );
         return;
      }

      if ( ww == charSize )
      {
         memcpy( result->data(), string->getRawStorage(), string->size() );
      }
      else
      {
         uint32 size = string->size();
         for( uint32 p = 0; p < size; p++ )
         {
            result->set( p, string->getCharAt(p) );
         }
      }
   }

   vm->retval( result );
}

/*# function strFromMemBuf
   @param membuf A MemBuf that will be converted to a string.
   @brief Convets a MemBuf to a string
   @return The resulting string
   
   This string takes each element of the membuf and converts it into
   a character in the final string. The contents of the buffer are
   not transcoded. It is appropriate to say that this function considers 
   each element in the MemBuf as an Unicode value for the character in the
   final string.

   To create a string from a buffer that may come from an encoded source
   (i.e. a file), use directly Transcode functions.
*/

FALCON_FUNC  strFromMemBuf ( ::Falcon::VMachine *vm )
{
   // Parameter checking;
   Item *i_membuf = vm->param(0);

   if( i_membuf == 0 || ! i_membuf->isMemBuf() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
           extra("M") ) );
      return;
   }

   MemBuf *mb = i_membuf->asMemBuf();

   // preallocating size instead of len, we won't have to resize the memory even
   // if resizing character sizes.
   String *result = new GarbageString( vm, mb->size() );

   uint32 size = mb->length();
   for( uint32 p = 0; p < size; p++ )
   {
      result->append( mb->get( p ) );
   }

   vm->retval( result );
}

}}


/* end of string.cpp */
