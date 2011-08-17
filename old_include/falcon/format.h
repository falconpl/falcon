/*
   FALCON - The Falcon Programming Language.
   FILE: format.h

   Format base class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab giu 16 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Format base class
*/

#ifndef flc_format_H
#define flc_format_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/falcondata.h>

/* To be added to docs in the next version.
convType
Type of conversion; determines what type of variable is expected when formatting, depending on the format string. for example, when a decimal number is indicated in the format string, it is implied that the input variable to be formatted is meant to be a numeric value.
decimalChr
Unicode character used to separate decimal parts of numbers. Defaults to '.'.
decimals
Number of decimals that should be represented. Zero means to represent only integer numbers.
fixedSize
If true, the 'size' field is mandatory, and representation of this variable is truncated to a maximum of 'size' characters.
grouping
Count of grouping characters in numeric representation. For example, a number like "1,000,000" has a grouping of 3, while Japanese stanrd representation is 4-grouping (like "10,0000"). Zero means no grouping.
groupingChr
Character using for grouping. Defaults to comma (',').
misAct
Action to perform when a variable of unexpected type is formatted. Can be:
*/

namespace Falcon {

class VMachine;
class Item;


/** Item to string format class. */

class Format: public FalconData
{

public:

   /** negative representation style */
   typedef enum {
      /** Optional minus in front of the number if negative. */
      e_minusFront,
      /** Plus and minus in front of the number. */
      e_plusMinusFront,
      /** Minus after the number if negative. */
      e_minusBack,
      /** Plus or minus after the number always. */
      e_plusMinusBack,
      /** Minus at field alignment end. */
      e_minusEnd,
      /** Plus or minus at field alignment end. */
      e_plusMinusEnd,
      /** In parenthesis if negative */
      e_parenthesis,
      /** In parenthesis if negative, add padding space if positive */
      e_parpad
   } t_negFormat;

   /** Integer representation style */
   typedef enum {
      /** No transformation */
      e_decimal,
      /** To binary */
      e_binary,
      /** To binary, add b after the number */
      e_binaryB,
      /** To octal */
      e_octal,
      /** To octal, add 0 in front */
      e_octalZero,
      /** To hex, lower case */
      e_hexLower,
      /** To hex, upper case */
      e_hexUpper,
      /** To hex, lowercase, add 0x in front */
      e_cHexLower,
      /** To hex, uppercase, add 0x in front */
      e_cHexUpper,
      /** To scientific notation. */
      e_scientific
   } t_numFormat;

   /** Conversion type */
   typedef enum {
      /** Basically a numeric conversion */
      e_tNum,
      /** Basically a string conversion */
      e_tStr,
      /** Wrong format / parse error */
      e_tError
   }
   t_convType;

   /** Action to perform on type mismatch / impossible conversion */
   typedef enum {
      /** Just return false. In the VM, this will cause the formatter to return nil,
          or to raise an error if embedded in strings. */
      e_actNoAction,
      /** Act as incompatible types were nil */
      e_actNil,
      /** Act as incompatible types were zero or empty */
      e_actZero,
      /** Raise a type error */
      e_actRaise,
      /** Try to convert, if conversion fails act as if nil */
      e_actConvertNil,
      /** Try to convert, if conversion fails act as if zero of empty */
      e_actConvertZero,
      /** Try to convert, if conversion fails raise a type error */
      e_actConvertRaise
   }
   t_convMismatch;

   /** How to represent a nil */
   typedef enum {
      /** Nil is not represented */
      e_nilEmpty,
      /** Nil is written as 'Nil' */
      e_nilNil,
      /** Nil is written as 'N' */
      e_nilN,
      /** Nil is written as 'nil' */
      e_nilnil,
      /** Nil is written as N/A */
      e_nilNA,
      /** Nil is written as None */
      e_nilNone,
      /** Nil is written as NULL */
      e_nilNULL,
      /** Nil is written as Null */
      e_nilNull,
      /** Nil is written as padding */
      e_nilPad
   }
   t_nilFormat;


private:
   t_convType m_convType;
   t_convMismatch m_misAct;
   uint16 m_size;

   uint32 m_paddingChr;
   uint32 m_thousandSep;
   uint32 m_decimalSep;

   uint8 m_grouping;
   bool m_fixedSize;
   t_nilFormat m_nilFormat;

   bool m_rightAlign;
   uint8 m_decimals;
   t_negFormat m_negFormat;
   t_numFormat m_numFormat;

   String m_originalFormat;
   uint32 m_posOfObjectFmt;

   void formatInt( int64 number, String &target, bool bUseGroup );
   void applyNeg( String &target, int64 number );
   void applyPad( String &target, uint32 extraSize=0 );

   /** Calculate needs of padding size. */
   int negPadSize( int64 number );

   /** Return true if negative format should be added before adding padding. */
   bool negBeforePad();

   /** Process a format mismatch. */
   bool processMismatch( VMachine *vm, const Item &source, String &target );

   /** Try a basic conversion into the desired item of this format. */
   bool tryConvertAndFormat( VMachine *vm, const Item &source, String &target );

   /** Procude a scientific string */
   void formatScientific( numeric num, String &sBuffer );

public:

   Format() {
      reset();
   }

   Format( const String &fmt )
   {
      reset();
      parse( fmt );
   }


   /** Parses a format string.
      Transforms a format string into a setup for this format object.

      The format is a sequence of commands that are parsed independently
      from their position. Commands are usually described by one, two or
      more character.

      Formats are meant to deal with different item types. A format thought
      for a certain kind of object, for example, a number, may be applied to something
      different, for example, a string, or the other way around.

      For this reason, Falcon formats include also what to do if the given item is
      not thought for the given format.

      Format elements:
      - Size: The minimum field lengt; it can be just expressed by a number. if the formatted
            output is wide as or wider than the allocated size, the output will NOT be truncated,
            and the resulting string may be just too wide to be displayed where it was intented
            to be. The size can be mandatory by adding '*' after it. In this case, the function
            will return false (and eventually raise an error) if the conversion caused the
            output to be wider than allowed.

      - Padding: the padding character is appended after the formatted size, or it is prepended
            before it alignment is right. To define padding character, use 'p'
            followed by the character. For example, p0 to fill the field with zeroes. Of course,
            the character may be any Unicode character (the format string accepts standard
            falcon character escapes). In the special case of p0, front sign indicators
            are placed at the beginning of the field; for example "4p0+" will produce "+001"
            "-002" and so on, while "4px+" will produce "xx+1", "xx-2" etc.

      - Numeric base: the way an integer should be rendered. It may be:
         - Decimal; as it's the default translation, no command is needed; a 'N' character may
               be added to the format to specify that we are actually expecting a number.
         - Hexadecimal: Command may be 'x' (lowercase hex), 'X' (uppercase Hex), 'c' (0x prefixed
               lowercase hex) or 'C' (0x prefixed uppercase hex).
         - Binary: 'b' to convert to binary, and 'B' to convert to binary and add a "b" after
                   the number.
         - Octal: 'o' to display an octal number, or '0' to display an octal with "0" prefix.
         - Scientific: 'e' to display a number in scientific notation W.D+/-eM.
               Format of numbers in scientific notation is fixed, so thousand separator
               and decimal digit separator cannot be set, but decimals cipher setting will
               still work.

      - Decimals: '.' followed by a number indicates the number of decimal to be displayed.
            If no decimal is specified, floating point numbers will be displayed with
            all significant digits digits, while if is's set to zero, decimal numbers
            will be rounded.

      - Decimal separator: a 'd' followed by any non-cipher character will be interpreted as
            decimal separator setting. For example, to use central european standard for
            decimal nubmers and limit the output to 3 decimals, write ".3d,", or "d,.3".
            The default value is '.'.

      - (Thousand) Grouping: actually it's the integer part group separator, as it will be
            displayed also for hexadecimal, octal and binary conversions. It is set using
            'g' followed by the separator character, it defaults to ','. Normally,
            it is not displayed; to activate it set also the integer grouping digit count;
            normally is 3, but it's 4 in Jpanaese and Chinese localses, while it may be useful
            to set it to 2 or 4 for hexadecimal, 3 for octal and 4 or 8 for binary. For example
            'g4-' would group digits 4 by 4, grouping them with a "-". Zero would disable
            grouping.

      - Grouping Character: If willing to change only the grouping character and not the default
            grouping count, use 'G'.

      - Alignment: by default the field is aligned to the left; to align the field to the right
            use 'r'.

      - Negative display format: By default, a '-' sign is appended in front of the number if
            it's negative. If the '+' character is added to the format, then in case the
            number is positive, '+' will be appended in front.
            '--' will postpend a '-' if the number is negative, while '++' will postpend either
            '+' or '-' depending on the sign of the number. To display a parenthesis around
            negative numbers, use '[', or use ']' to display a parenthesis for negative numbers
            and use the padding character in front and after positive numbers. Using parenthesis
            will prevent using '+', '++' or '--' formats. Format '-^' will add a - in front of
            padding space if the number is negative, while '+^' will add plus or minus depending
            on number sign. For example, "5+" would render -12 as "  -12", while "5+^" will render
            as "-  12". If alignment is to the right, the sign will be added at the other side
            of the padding: "5+^r" would render -12 as "12  -".
            If size is not mandatory, parenthesis
            will be wrapped around the formatted field, while if size is mandatory they will be
            wrapped around the whole field, included padding. For example "5[r" on -4 would render
            as "  (4)", while "5*[r" would render as "(  4)".

      - Object specific format: Objects may accept an object specific formatting as parameter
         of the standard "toString" method. A pipe separator '|' will cause all the following
         format to be passed unparsed to the toString method of objects eventually being
         formatted. If the object does not provides a toString method, or if it's not an
         object at all, an error will be raised. The object is the sole responsible for
         parsing and applying its specific format.

      - Nil format: How to represent a nil. It may be one of the following:
         - 'nn': nil is not represented (mute).
         - 'nN': nil is represented by "N"
         - 'nl': nil is rendered with "nil"
         - 'nL': nil is rendered with "Nil". This is also the default.
         - 'nu': nil is rendered with "Null"
         - 'nU': nil is rendered with "NULL"
         - 'no': nil is rendered with "None"
         - 'nA': nil is rendered with "NA"

      - Action on error: Normally, if trying to format something different from
            what is expected, format() the method will simply return false. For example,
            to format a string in a number, a string using the date formatter, a number in
            a simple pad-and-size formatter etc. To change this
            behavior, use '/' followed by one of the following:
         - 'n': act as the wrong item was nil (and uses the defined nil formatter).
         - '0': act as if the given item was 0, the empty string or an invalid date, or anyhow
                the neuter member of the expected type.
         - 'r': raise a type error.
         A 'c' letter may be added after the '/' and before the specifier to try a basic conversion
               into the expected type before triggering the requested effect. This will, for example,
               cause the toString() method of objects to be called if the formatting is detected
               to be a string-type format.

      If the pattern is invalid, a paramter error will be raised.
      Examples:
         - "8*Xs-g2": Mandatory 8 characters, Hexadecimal uppercase, grouped 2 by 2
            with '-' characters. A result may be "0A-F1-DA".
         - "12.3'0r+/r" - 12 ciphers, of which 3 are fixed decimals, 0 padded, right
            aligned. + is always added in front of positive numbers. In case the formatted
            item is not a number, a type error is raised.


      \note ranges will be represented as [n1:n2] or [n1:] if they are open. Size, alignment
         and padding will work on the whole range, while numeric formatting will be applied to
         each end of the range.

      \return true if parse is succesful, false on parse format error.
   */
   bool parse( const String &fmt );

   /** Checks if this format is valid. */
   bool isValid() const { return m_convType != e_tError; }


   /** Formats given item into the target string.
      This version doesn't require a VM, but will fail if source item
      is an object.

      Also, in case of failure there will be no report about the error
      context, as that would be normally reported by raising an exception
      in the VM.
      \see bool format( VMachine *vm, const Item &source, String &target )
      \param source the item to be formatted
      \param target the string where to output the format.
      \return true on success, false on error.
   */
   bool format( const Item &source, String &target ) {
      return format( 0, source, target );
   }

   /** Formats given item into the target string.
      The target string is not cleared before format occours; in other words,
      the format output is just appended to the end of the string. Use a
      pre-allocated string for better performance.

      In case the source item is an object, its toString() method is called
      on the given VM. In case of raise from the routine, the method will
      return false; if raise on failure is set, then the exception is left
      untouched and it will be passed to the calling routine, else the exception
      will be reset and the format routine will simply fail.

      \see parse for format details.

      \return true on success, false on error.
   */
   bool format( VMachine *vm, const Item &source, String &target );

   /** sets default values */
   void reset();

   //==============================================================
   // Accessors
   //

   const String &originalFormat() const { return m_originalFormat; }

   t_convType convType() const { return m_convType; }

   t_convMismatch mismatchAction() const { return m_misAct; }
   void mismatchAction( t_convMismatch t ) { m_misAct = t; }

   uint16 fieldSize() const { return m_size; }
   void fieldSize( uint16 size ) { m_size = size; }

   uint32 paddingChr() const { return m_paddingChr; }
   void paddingChr( uint32 chr ) { m_paddingChr = chr; }

   bool rightAlign() const { return m_rightAlign; }
   void rightAlign( bool b ) { m_rightAlign = b; }

   uint8 decimals() const { return m_decimals; }
   void decimals( uint8 d ) { m_decimals = d; }

   uint8 grouping() const { return m_grouping; }
   void grouping( uint8 g ) { m_grouping = g; }

   t_negFormat negFormat() const { return m_negFormat; }
   void negFormat( t_negFormat f ) { m_negFormat = f; }

   t_nilFormat nilFormat() const { return m_nilFormat; }
   void nilFormat( t_nilFormat f ) { m_nilFormat = f; }

   t_numFormat numFormat() const { return m_numFormat; }
   void numFormat( t_numFormat t ) { m_numFormat = t; }

   bool fixedSize() const { return m_fixedSize; }
   void fixedSize( bool f ) { m_fixedSize = f; }

   //=================================================

   virtual Format *clone() const;
   virtual void gcMark( uint32 mark ) {}
};

}

#endif

/* end of format.h */
