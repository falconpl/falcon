/*
   FALCON - The Falcon Programming Language
   FILE: format.cpp

   Format base class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar apr 17 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Format base class
*/

#include <falcon/format.h>
#include <falcon/vm.h>
#include <falcon/item.h>
#include <falcon/string.h>
#include <falcon/timestamp.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

namespace Falcon {

void Format::reset()
{
   m_convType = e_tStr;
   m_misAct = e_actNoAction;
   m_fixedSize = false;
   m_size =  0;
   m_paddingChr = ' ';
   m_thousandSep = ',';
   m_decimalSep = '.';
   m_grouping = 0;
   m_rightAlign = false;
   m_decimals = 0;
   m_negFormat = e_minusFront;
   m_numFormat = e_decimal;
   m_nilFormat = e_nilNil;
   m_posOfObjectFmt = String::npos;
}


bool Format::parse( const String &fmt )
{

   String tmp;
   uint32 pos = 0;
   uint32 len = fmt.length();

   typedef enum {
      e_sInitial,
      e_sSize,
      e_sDecimals,
      e_sPadding,
      e_sDecSep,
      e_sGroupSep,
      e_sGroupSep2,
      e_sErrorEffect,
      e_sErrorEffect2,
      e_sNilMode,
      e_sNegFmt,
      e_sNegFmt2
   }
   t_state;
   t_state state = e_sInitial;


   while( pos < len )
   {
      uint32 chr = fmt.getCharAt( pos );

      switch( state )
      {

         //=============================
         // Basic state.
         //
         case e_sInitial:
            if( chr >= '0' && chr <= '9' )
            {
               // size already given
               if ( m_size != 0 )
                  return false;

               state = e_sSize;
               tmp.size(0);
               tmp += chr;
               break;
            }

            // else:
            switch( chr )
            {
               case 'N':
                  // it should be an octal.
                  m_convType = e_tNum;
                  numFormat( e_decimal );
               break;

               case '.':
                  // it should be an octal.
                  m_convType = e_tNum;
                  state = e_sDecimals;
                  tmp.size(0);
               break;

               case 'b':
                  // it should be an octal.
                  m_convType = e_tNum;
                  numFormat( e_binary );
               break;

               case 'B':
                  // it should be an octal.
                  m_convType = e_tNum;
                  numFormat( e_binaryB );
               break;

               case 'd':
                  m_convType = e_tNum;
                  state = e_sDecSep;
               break;

               case 'p':
                  state = e_sPadding;
               break;

               case 'g':
                  m_convType = e_tNum;
                  state = e_sGroupSep;
               break;

               case 'G':
                  m_convType = e_tNum;
                  state = e_sGroupSep2;
               break;

               case '0':
                  // it should be an octal.
                  m_convType = e_tNum;
                  numFormat( e_octalZero );
               break;

               case 'o':
                  // it should be an octal.
                  m_convType = e_tNum;
                  numFormat( e_octal );
               break;

               case 'x':
                  // it should be an octal.
                  m_convType = e_tNum;
                  numFormat( e_hexLower );
               break;

               case 'X':
                  // it should be an octal.
                  m_convType = e_tNum;
                  numFormat( e_hexUpper );
               break;

               case 'c':
                  // it should be an octal.
                  m_convType = e_tNum;
                  numFormat( e_cHexLower );
               break;

               case 'C':
                  // it should be an octal.
                  m_convType = e_tNum;
                  numFormat( e_cHexUpper );
               break;

               case 'e':
                  // it should be in scientific format
                  m_convType = e_tNum;
                  numFormat( e_scientific );
               break;

               case '/':
                  // it should be an octal.
                  state = e_sErrorEffect;
               break;

               case 'n':
                  state = e_sNilMode;
               break;

               case '|':
                  m_posOfObjectFmt = pos;
                  m_convType = e_tStr;
                  // complete parsing
                  pos = len;
               break;

               case '+':
                  m_negFormat = e_plusMinusFront;
                  state = e_sNegFmt;
               break;

               case '-':
                  m_negFormat = e_minusFront;
                  state = e_sNegFmt2;
               break;

               case '[':
                  m_negFormat = e_parenthesis;
               break;

               case ']':
                  m_negFormat = e_parpad;
               break;

               case 'r':
                  m_rightAlign = true;
               break;

               default:
                  // unrecognized character
                  m_convType = e_tError;
                  return false;
            }
         break;

         //=============================
         // Parse padding
         //
         case e_sDecSep:
            m_decimalSep = chr;
            state = e_sInitial;
         break;

         case e_sPadding:
            m_paddingChr = chr;
            state = e_sInitial;
         break;

         case e_sGroupSep:
            if( chr >= '0' && chr <='9' )
            {
               m_grouping = chr - '0';
               state = e_sGroupSep2;
            }
            else {
               m_thousandSep = chr;
               state = e_sInitial;
            }
         break;

         case e_sGroupSep2:
            m_thousandSep = chr;
            state = e_sInitial;
         break;

         //=============================
         // Size parsing state
         //
         case e_sSize:
            if( chr >= '0' && chr <= '9' )
            {
               tmp += chr;

               // size too wide
               if ( tmp.length() > 4 ) {
                  m_convType = e_tError;
                  return false;
               }
            }
            else
            {
               int64 tgt;
               tmp.parseInt( tgt );
               fieldSize( (uint16) tgt );

               if( chr == '*' )
               {
                  fixedSize( true );
               }
               else {
                  // reparse current char
                  --pos;
               }

               state = e_sInitial;
            }
         break;

         //=============================
         // Decimals parsing state
         //
         case e_sDecimals:
            if( chr >= '0' && chr <= '9' )
            {
               tmp += chr;

               // size too wide
               if ( tmp.length() > 2 ) {
                  m_convType = e_tError;
                  return false;
               }
            }
            else
            {
               int64 tgt;
               tmp.parseInt( tgt );
               decimals( (uint8) tgt );
               // reparse current char
               --pos;
               state = e_sInitial;
            }
         break;

         //===============================================
         // Parsing what should be done in case of error.
         //
         case e_sErrorEffect:
            if ( chr == 'c' )
            {
               state = e_sErrorEffect2;
               break;
            }

            // else
            switch( chr )
            {
               case 'n': mismatchAction( e_actNil ); break;
               case '0': mismatchAction( e_actZero ); break;
               case 'r': mismatchAction( e_actRaise ); break;

               default:
                  // invalid choiche
                  m_convType = e_tError;
                  return false;
            }

            state = e_sInitial;
         break;

         case e_sErrorEffect2:
            switch( chr )
            {
               case 'n': mismatchAction( e_actConvertNil ); break;
               case '0': mismatchAction( e_actConvertZero ); break;
               case 'r': mismatchAction( e_actConvertRaise ); break;

               default:
                  // invalid choiche
                  m_convType = e_tError;
                  return false;
            }
            state = e_sInitial;
         break;

         //=================================
         // parsing what do to with a Nil
         //
         case e_sNilMode:
            switch( chr )
            {
               case 'n': m_nilFormat = e_nilEmpty; break;
               case 'N': m_nilFormat = e_nilN; break;
               case 'l': m_nilFormat = e_nilnil; break;
               case 'L': m_nilFormat = e_nilNil; break;
               case 'u': m_nilFormat = e_nilNull; break;
               case 'U': m_nilFormat = e_nilNULL; break;
               case 'o': m_nilFormat = e_nilNone; break;
               case 'A': m_nilFormat = e_nilNA; break;

               default:
                  m_convType = e_tError;
                  return false;
            }
            state = e_sInitial;
         break;

         //=================================
         // Parsing neg format
         case e_sNegFmt:
            switch( chr ) {
               case '+': m_negFormat = e_plusMinusBack; break;
               case '^': m_negFormat = e_plusMinusEnd; break;
               default:
                  pos--;
            }
            state = e_sInitial;
         break;

         //=================================
         // Parsing neg format 2
         case e_sNegFmt2:
            switch( chr ) {
               case '-': m_negFormat = e_minusBack; break;
               case '^': m_negFormat = e_minusEnd; break;
               default:
                  pos--;
            }
            state = e_sInitial;
         break;

      }

      ++pos;
   } // end main loop


   // verify output status
   switch( state )
   {
      case e_sInitial: // ok
      case e_sNegFmt:
      break;

      case e_sSize:
      {
         int64 tgt;
         tmp.parseInt( tgt );
         fieldSize( (uint8) tgt );
      }
      break;

      case e_sDecimals:
      {
         int64 tgt;
         tmp.parseInt( tgt );
         decimals( (uint8) tgt );
      }
      break;

      // any other state means we're left in the middle of something
      default:
         m_convType = e_tError;
         return false;
   }

   // if everything goes fine...
   m_originalFormat = fmt;
   return true;
}


bool Format::format( VMachine *vm, const Item &source, String &target )
{
   String sBuffer;

   switch( source.type() )
   {
      case FLC_ITEM_NIL:
         switch( m_nilFormat )
         {
            case e_nilEmpty: break;
            case e_nilNil: sBuffer = "Nil"; break;
            case e_nilN: sBuffer = "N"; break;
            case e_nilnil: sBuffer = "nil"; break;
            case e_nilNA: sBuffer = "N/A"; break;
            case e_nilNone: sBuffer = "None"; break;
            case e_nilNULL: sBuffer = "NULL"; break;
            case e_nilNull: sBuffer = "Null"; break;
            case e_nilPad: sBuffer.append( m_paddingChr );
         }
         applyPad( sBuffer );

      break;

      case FLC_ITEM_UNB:
         sBuffer = "_";
         applyPad( sBuffer );
         break;


      //==================================================
      // Parse an integer
      //
      case FLC_ITEM_INT:
      {
         int64 num = source.asInteger();

         // number formats are compatible with string formats
         if ( m_convType != e_tNum && m_convType != e_tStr )
         {
            return processMismatch( vm, source, target );
         }

         formatInt( num, sBuffer, true );

         // minus sign must be added AFTER padding with parentesis/fixed size or with *End signs,
         // else it must be added before.
         if ( negBeforePad() )
         {
            applyNeg( sBuffer, num );
            applyPad( sBuffer );
         }
         else {
            applyPad( sBuffer, negPadSize( num ) );
            applyNeg( sBuffer, num );
         }
      }
      break;

      //==================================================
      // Parse double format
      //
      case FLC_ITEM_NUM:
      {
         numeric num = source.asNumeric();

         // number formats are compatible with string formats
         if ( m_convType != e_tNum && m_convType != e_tStr )
         {
            return processMismatch( vm, source, target );
         }

         if( m_numFormat == e_scientific )
         {
           formatScientific( num, sBuffer );
         }
         else {
            double intPart, fractPart;
            bool bNeg, bIntIsZero;
            fractPart = modf( num, &intPart );
            if ( intPart < 0.0 ) {
               intPart = -intPart;
               fractPart = -fractPart;
               bNeg = true;
               bIntIsZero = false;
            }
            else
            {
               bIntIsZero = intPart > 0.0 ? false : true;

               if ( fractPart < 0.0 )
               {
                  fractPart = -fractPart;
                  // draw neg sign only if < 0 but int
                  bNeg = true;
               }
               else
                  bNeg = false;
            }



            String precPart;
            int base = 10;
            switch( m_numFormat )
            {
               case e_binary: case e_binaryB: base = 2; break;
               case e_octalZero: case e_octal: base = 8; break;
               case e_cHexUpper: case e_hexUpper: case e_cHexLower: case e_hexLower: base = 16; break;
               default:
                  break;
            }

            while( intPart > 9e14 )
            {
               intPart /= base;
               precPart.append( '0' );
            }

            // manual round
            if( m_decimals == 0 && fractPart >= 0.5 )
            {
               intPart++;
               bIntIsZero = false;
            }

            uint8 decs = m_decimals;
            m_decimals = 0;


            formatInt( (int64) intPart, sBuffer, false );
            sBuffer.append( precPart );

            // now we can add the grouping
            if ( m_grouping > 0 )
            {
               String token;
               token.append( m_thousandSep );
               uint32 pos = sBuffer.size();

               while( pos > m_grouping )
               {
                  pos -= m_grouping;
                  sBuffer.insert( pos, 0, token );
               }
            }

            // finally add decimals
            m_decimals = decs;
            if( base == 10 && m_decimals > 0 )
            {

               char bufFmt[32];
               char buffer[255];
               sprintf( bufFmt, "%%.%df", m_decimals );
               sprintf( buffer, bufFmt, fractPart );
               sBuffer.append( m_decimalSep );
               sBuffer.append( buffer + 2 );
            }
            else if ( bIntIsZero )
            {
               // do not print -0!
               bNeg = false;
            }

            // we must fix the number.
            num = bNeg ? -1.0 : 1.0;
         }

         // minus sign must be added AFTER padding with parentesis/fixed size or with *End signs,
         // else it must be added before.
         if ( negBeforePad() )
         {
            applyNeg( sBuffer, (int64) num );
            applyPad( sBuffer );
         }
         else {
            applyPad( sBuffer, negPadSize( (int64) num ) );
            applyNeg( sBuffer, (int64) num );
         }
      }
      break;

      case FLC_ITEM_RANGE:
      {
         // number formats are compatible with string formats
         if ( m_convType != e_tNum && m_convType != e_tStr )
         {
            return processMismatch( vm, source, target );
         }

         int64 begin = source.asRangeStart();
         String sBuf1, sBuf2, sBuf3;

         formatInt( begin, sBuf1, true );

         //apply negative format now.
         applyNeg( sBuf1, (int64) begin );
         if ( ! source.asRangeIsOpen() )
         {
            int64 end = source.asRangeEnd();
            formatInt( end, sBuf2, true );
            applyNeg( sBuf2, (int64) end );
            
            int64 step = source.asRangeStep();
            if ( (begin <= end && step != 1) ||
                 (begin > end && step != -1 ) )
            {
               formatInt( step, sBuf3, true );
               applyNeg( sBuf3, (int64) step );
               sBuffer = "[" + sBuf1 + ":" + sBuf2 + ":" + sBuf3 + "]";
            }
            else
               sBuffer = "[" + sBuf1 + ":" + sBuf2 + "]";
         }
         else
            sBuffer = "[" + sBuf1 + ":" + sBuf2 + "]";

         applyPad( sBuffer );
      }
      break;

      case FLC_ITEM_STRING:
      {
         // number formats are compatible with string formats
         if ( m_convType != e_tStr )
         {
            return processMismatch( vm, source, target );
         }

         sBuffer = *source.asString();
         applyPad( sBuffer );
      }
      break;


      case FLC_ITEM_OBJECT:
      {
         // try to format the object
         if( vm != 0 )
         {
            if( m_posOfObjectFmt != String::npos )
            {
               vm->itemToString( sBuffer, &source, m_originalFormat.subString( m_posOfObjectFmt + 1 ) );
            }
            else {
               vm->itemToString( sBuffer, &source );
            }
         }
         else {
            return processMismatch( vm, source, target );
         }

         applyPad( sBuffer );
      }
      break;

      default:
         return processMismatch( vm, source, target );
   }

   // out of bounds?
   if ( m_size > 0 && m_fixedSize && sBuffer.length() > m_size ) {
      return false;
   }

   target += sBuffer;
   return true;
}



void Format::formatScientific( numeric num, String &sBuffer )
{
   char buffer[36];
   bool bNeg;
   if( num < 0 )
   {
      num = - num;
      bNeg = true;
   }
   else {
      bNeg = false;
   }
   sprintf( buffer, "%35e", num );
   sBuffer += buffer;
}


bool Format::processMismatch( VMachine *vm, const Item &source, String &target )
{
   Item dummy;

   switch( m_misAct )
   {
      case e_actNoAction:
         return false;

      case e_actConvertNil:
         if( tryConvertAndFormat( vm, source, target ) )
            return true;

         // else fallthrouhg
      case e_actNil:
         dummy.setNil();
         return format( vm, dummy, target );
      break;

      case e_actConvertZero:
         if( tryConvertAndFormat( vm, source, target ) )
            return true;

         // else fallthrouhg
      case e_actZero:
      {
         dummy = (int64) 0;

         // also, force conversion
         m_misAct = e_actConvertZero;
         bool ret = format( vm, source, target );
         m_misAct = e_actZero;
         return ret;
      }

      case e_actConvertRaise:
         if( tryConvertAndFormat( vm, source, target ) )
            return true;

         // else fallthrouhg
      case e_actRaise:
         if( vm != 0 )
         {
            throw new TypeError( ErrorParam( e_fmt_convert, __LINE__ )
               .origin( e_orig_runtime ) );
         }
         return false;
   }

   // should not happen
   return false;
}

bool Format::tryConvertAndFormat( VMachine *vm, const Item &source, String &target )
{
   // first convert to string, then try to convert to number

   // try a basic string conversion
   String temp;
   if( vm != 0 )
   {
      if( m_posOfObjectFmt != String::npos )
      {
         vm->itemToString( temp, &source, originalFormat().subString( m_posOfObjectFmt + 1 ) );
      }
      else
         vm->itemToString( temp, &source );
   }
   else {
      source.toString( temp );
   }

   // If conversion was numeric, try to to reformat the thing into a number
   if( m_convType == e_tNum )
   {
      numeric num;
      if( ! temp.parseDouble( num ) )
         return false;

      return format( vm, num, target );
   }

   return format( vm, &temp, target );
}


void Format::formatInt( int64 number, String &target, bool bUseGroup )
{

   if ( m_numFormat == e_scientific )
   {
      formatScientific( (numeric) number, target );
      return;
   }

   if ( number == 0 )
   {
      target += "0";
      return;
   }

   // prepare the buffer
   const int bufSize = 132; // int64 binary size + separator per each bit + minus format + zero
   uint32 buffer[ bufSize ];

   uint32 pos = bufSize - 2;
   buffer[ pos+1 ] = 0; // allow room for post formatters


   if ( number < 0 )
   {
      number = - number;
   }

   // init grouping
   int grp = m_grouping;

   // init base
   int base = 10;
   uint32 baseHexChr = (uint32) 'X';
   switch( m_numFormat )
   {
      case e_decimal: base = 10; break;
      case e_binary: case e_binaryB: base = 2; break;

      case e_octalZero:
         target.append( '0' );
         // fallthrough
      case e_octal:
         base = 8;
      break;

      case e_cHexLower:
         target.append( '0' );
         target.append( 'x' );
         // fallthrough
      case e_hexLower:
         base = 16;
         baseHexChr = 'a';
      break;

      case e_cHexUpper:
         target.append( '0' );
         target.append( 'x' );
         // fallthrough
      case e_hexUpper:
         base = 16;
         baseHexChr = 'A';
      break;

      default:
         break;
   }

   while( number != 0 )
   {
      uint32 cipher =(uint32) (number % base);
      if ( cipher < 10 )
      {
         buffer[pos--] = (char) ( cipher + 0x30 );
      }
      else
      {
         buffer[pos--] = (char) ( (cipher-10) + baseHexChr );
      }
      number /= base;

      if( number != 0 && bUseGroup && m_grouping != 0 )
      {
         if( --grp == 0 ) {
            buffer[pos--] = (char) m_thousandSep;
            grp = m_grouping;
         }
      }
   }

   pos++;

   // unroll the parsed buffer
   while( buffer[pos] != 0 )
   {
      target += buffer[pos];
      ++pos;
   }

   if( m_numFormat == e_decimal )
   {
      if( m_decimals != 0 )
      {
         target.append( m_decimalSep );
         for( int d = 0; d < m_decimals; d ++ )
            target.append( '0' );
      }
   }
   else if ( m_numFormat == e_binaryB )
   {
      target.append( 'b' );
   }
}


void Format::applyNeg( String &target, int64 number )
{
   // apply negative format
   if ( number < 0 )
   {
      switch( m_negFormat )
      {
         case e_plusMinusFront:
         case e_minusFront:
            target.prepend( '-' );
         break;

         case e_plusMinusBack:
         case e_minusBack:
            target.append( '-' );
         break;

         case e_minusEnd:
         case e_plusMinusEnd:
            if( m_rightAlign )
               target.prepend( '-' );
            else
               target.append( '-' );
         break;

         case e_parpad:
         case e_parenthesis:
            target.append( ')' );
            target.prepend( '(' );
         break;
      }
   }
   else
   {
      switch( m_negFormat )
      {
         case e_plusMinusFront:
            target.prepend( '+' );
         break;

         case e_plusMinusBack:
            target.append( '+' );
         break;

         case e_plusMinusEnd:
            if( m_rightAlign )
               target.prepend( '+' );
            else
               target.append( '+' );
         break;

         case e_parpad:
            target.prepend( m_paddingChr );
            target.append( m_paddingChr );
         break;

         default:
            break;
      }
   }

}

int Format::negPadSize( int64 number )
{
   if ( number < 0 )
   {
      switch( m_negFormat )
      {
         case e_minusFront: return 1;
         case e_plusMinusFront: return 1;
         case e_minusBack: return 1;
         case e_plusMinusBack: return 1;
         case e_parenthesis: return 2;
         case e_parpad: return 2;
         case e_minusEnd: return 1;
         case e_plusMinusEnd: return 1;
      }
   }
   else
   {
      switch( m_negFormat )
      {
         case e_minusFront: return 0;
         case e_plusMinusFront: return 1;
         case e_minusBack: return 0;
         case e_plusMinusBack: return 1;
         case e_parenthesis: return 0;
         case e_parpad: return 2;
         case e_minusEnd: return 0;
         case e_plusMinusEnd: return 1;
      }
   }

   // should never happen
   return 0;
}

bool Format::negBeforePad()
{
   // if the format requires it explicitly...
   if ( m_negFormat == e_plusMinusEnd || m_negFormat == e_minusEnd )
      return false;

   // else, if padding char is 0...
   if ( m_paddingChr == '0' )
      return false;

   // finally, if size is fixed and parenthesis are requested
   if ( m_fixedSize && ( m_negFormat == e_parenthesis || m_negFormat == e_parpad ) )
      return false;

   return true;
}

void Format::applyPad( String &target, uint32 extraSize )
{
   uint32 tgSize = m_size;
   uint32 strSize = target.length() + extraSize;

   if( tgSize <= strSize )
      return;

   String strBuffer;
   strBuffer.reserve( tgSize - strSize );
   while( strSize < tgSize )
   {
      strBuffer.append( m_paddingChr );
      ++strSize;
   }

   if( m_rightAlign )
   {
      target.prepend( strBuffer );
   }
   else {
      target += strBuffer;
   }
}

/*

void Format::getProperty( const String &propName, Item &prop )
{
   if( propName == "size" )
   {
      prop = (int64) m_size;
   }
   else if( propName == "decimals" ) {
      prop = (int64) m_decimals;
   }
   else if( propName == "paddingChr" ) {
      prop = (int64) m_paddingChr;
   }
   else if( propName == "groupingChr" ) {
      prop = (int64) m_thousandSep;
   }
   else if( propName == "decimalChr" ) {
      prop = (int64) m_decimalSep;
   }
   else if( propName == "grouiping" ) {
      prop = (int64) m_grouping;
   }
   else if( propName == "fixedSize" ) {
      prop = (int64) (m_fixedSize ? 1:0);
   }
   else if( propName == "rightAlign" ) {
      prop = (int64) (m_rightAlign ? 1:0);
   }
   else if( propName == "originalFormat" ) {
      prop = &m_originalFormat;
   }
   else if( propName == "convType" ) {
      prop = (int64) m_convType;
   }
   else if( propName == "misAct" ) {
      prop = (int64) m_misAct;
   }
   else if( propName == "nilFormat" ) {
      prop = (int64) m_nilFormat;
   }
   else if( propName == "negFormat" ) {
      prop = (int64) m_negFormat;
   }
   else if( propName == "numFormat" ) {
      prop = (int64) m_numFormat;
   }

}

void Format::setProperty( const String &propName, Item &prop )
{
   // read only
}
*/

Format *Format::clone() const
{
   return new Format( this->originalFormat() );
}

}

/* end of format.cpp */
