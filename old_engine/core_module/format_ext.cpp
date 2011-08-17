/*
   FALCON - The Falcon Programming Language.
   FILE: format_ext.cpp

   The Format pretty print formatter class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 02:06:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"
#include <falcon/format.h>
#include <falcon/falconobject.h>
#include <falcon/fassert.h>

/*#
   @beginmodule core
*/

namespace Falcon {
namespace core {

/*#
   @class Format
   @brief Controls the rendering of items into strings.
   @ingroup general_purpose
   @optparam fmtspec If provided, must be a valid format specifier
      which is immediately parsed. In case of invalid format,
      a ParseError is raised.

   Format class is meant to provide an efficient way to format variables into
   strings that can then be sent to output streams. Internally, the format class
   is used in string expansion (the '\@' operator), but while string expansion causes
   a string parsing to be initiated and an internal temporary Format object to be
   instantiated each time an expansion is performed, using a prebuilt Format object
   allows to optimize repeated formatting operations. Also, Format class instances may
   be used as other objects properties, applied directly to strings being written on streams,
   modified after being created and are generally more flexible than the string expansion.

   The format specifier is a string that may contain various elements indicating how the target
   variable should be rendered as a string.

   @section format_specification Format specification

      @b Size: The minimum field length; it can be just expressed by a number.
      if the formatted output is wide as or wider than the allocated size, the output
      will not be truncated, and the resulting string may be just too wide to be displayed
      where it was intended to be. The size can be mandatory by adding '*' after it.
      In this case, the format() method will return false (and eventually raise an error)
      if the conversion caused the output to be wider than allowed.

      @b Padding: the padding character is appended after the formatted size, or it is put in
      front of it if alignment is to the right. To define padding character, use 'p' followed
      by the character. In example, p0 to fill the field with zeros. Of course, the character
      may be any Unicode character (the format string accepts standard Falcon character escapes).
      In the special case of p0, front sign indicators are placed at the beginning of the field;
      in example "4p0+" will produce "+001" "-002" and so on, while "4px+" will produce "xx+1", "xx-2" etc.

      @b Numeric @b base: the way an integer should be rendered. It may be:
         - Decimal: as it's the default translation, no command is needed; a 'N' character may
           be added to the format to specify that we are actually expecting a number.
         - Hexadecimal: Command may be 'x' (lowercase hex), 'X' (uppercase Hex), 'c'
           (0x prefixed lowercase hex) or 'C' (0x prefixed uppercase hex). Binary: 'b' to
           convert to binary, and 'B' to convert to binary and add a "b" after the number.

         - Octal: 'o' to display an octal number, or '0' to display an octal with "0" prefix.
         - Scientific: 'e' to display a number in scientific notation W.D+/-eM. Format of numbers in
           scientific notation is fixed, so thousand separator and decimal digit separator cannot be set,
           but decimals cipher setting will still work.

      @b Decimals: a dot '.' followed by a number indicates the number of decimal to be displayed. If no
          decimal is specified, floating point numbers will be displayed with all significant digits digits,
          while if it's set to zero, decimal numbers will be rounded.

      @b Decimal @b separator: a 'd' followed by any non-cipher character will be interpreted as decimal
      separator setting. For example, to use central European standard for decimal nubmers and limit the
      output to 3 decimals, write ".3d,", or "d,.3". The default value is '.'.

      @b (Thousands) @b Grouping: actually it's the integer part group separator, as it will be displayed
      also for hexadecimal, octal and binary conversions. It is set using 'g' followed by the separator
      character, it defaults to ','. Normally, it is not displayed; to activate it set also the integer
      grouping digit count; normally is 3, but it's 4 in Japanaese and Chinese locales, while it may be
      useful to set it to 2 or 4 for hexadecimal, 3 for octal and 4 or 8 for binary. In example 'g4-'
      would group digits 4 by 4, grouping them with a "-". Zero would disable grouping.

      @b Grouping @b Character: If willing to change only the grouping character and not the default
      grouping count, use 'G'.

      @b Alignment: by default the field is aligned to the left; to align the field to
         the right use 'r'.
      @b Negative @b display @b format: By default, a '-' sign is appended in front of the number if it's
         negative. If the '+' character is added to the format, then in case the number is positive, '+'
         will be appended in front. '--' will postpend a '-' if the number is negative, while '++'
         will postpend either '+' or '-' depending on the sign of the number. To display a parenthesis around
         negative numbers, use '[', or use ']' to display a parenthesis for negative numbers and use the padding
         character in front and after positive numbers. Using parenthesis will prevent using '+', '++' or '--'
         formats. Format '-^' will add a - in front of padding space if the number is negative, while '+^'
         will add plus or minus depending on number sign. In example, "5+" would render -12 as "  -12", while "5+^"
          will render as "-  12". If alignment is to the right, the sign will be added at the other side of the
          padding: "5+^r" would render -12 as "12  -". If size is not mandatory, parenthesis will be wrapped
          around the formatted field, while if size is mandatory they will be wrapped around the whole field,
          included padding. In example "5[r" on -4 would render
          as "  (4)", while "5*[r" would render as "(  4)".

      @b Nil @b format: How to represent a nil. It may be one of the following:
         - 'nn': nil is not represented (mute).
         - 'nN': nil is represented by "N"
         - 'nl': nil is rendered with "nil"
         - 'nL': nil is rendered with "Nil". This is also the default.
         - 'nu': nil is rendered with "Null"
         - 'nU': nil is rendered with "NULL"
         - 'no': nil is rendered with "None"
         - 'nA': nil is rendered with "NA"

      @b Action @b on @b error: Normally, if trying to format something different from
      what is expected, the method format() will simply return false. In example, to format
      a string in a number, a string using the date formatter, a number in a simple
      pad-and-size formatter etc. To change this behavior, use '/' followed by one of
      the following:
         - 'n': act as the wrong item was nil (and uses the defined nil formatter).
         - '0': act as if the given item was 0, the empty string or an invalid date,
                or anyhow the neuter item of the expected type.
         - 'r': raise a type error.

      A 'c' letter may be added after the '/' and before the specifier to try a
      basic conversion into the expected type before triggering the requested effect.
      In example, if the formatted item is an object and the conversion type is string
      (that is, no numeric related options are set), this will cause the toString()
      method of the target object to be called, or if not available, the toString()
      function to be applied on the target object. In example "6/cr" tries to convert the
      item to a 6 character long string, and if it fails (i.e. because toString() method
      returns nil) an TypeError is raised.

      @b Object @b specific @b format: Objects may accept an object specific formatting as
      parameter of the standard toString() method. A pipe separator '|' will cause all the
      following format to be passed unparsed to the toString() method of objects eventually
      being formatted. If the object does not provides a toString() method, or if it's not
      an object at all, an error will be raised. The object is the sole responsible for
      parsing and applying its specific format.

      @note Ranges will be represented as [n1:n2:n3] or [n1:] if they are open. Size, alignment and padding
      will work on the whole range, while numeric formatting will be applied to each end of the range.

      Example: the format specifier "8*Xs-g2" means to format variables in a field
      of 8 characters, size mandatory (i.e. truncated if wider), Hexadecimal uppercase, grouped 2 by 2
      with '-' characters. A result may be "0A-F1-DA".

      Another example: "12.3'0r+/r" means to format a number in 12 ciphers, of which 3 are
      fixed decimals, 0 padded, right aligned; a '+' is always added in front of positive
      numbers. In case the formatted item is not a number, a type error is raised.

      Format class instances may be applied on several variables; in example, a currency value
      oriented numeric format may be applied on all the currency values of a program, and changing
      the default format would just be a matter of changing just one format object.
*/

/*#
   @method parse Format
   @brief Initializes the Format instance with an optional value.
   @param fmtspec Format specifier
   @raise ParseError if the format specifier is not correct.

   Sets or changes the format specifier for this Format instance.
   If the format string is not correct, a ParseError is raised.
*/
FALCON_FUNC  Format_parse ( ::Falcon::VMachine *vm )
{

   CoreObject *einst = vm->self().asObject();
   Format *fmt = (Format *) einst->getFalconData();

   Item *param = vm->param( 0 );
   if ( param != 0 )
   {
      if( ! param->isString() )
      {
         throw new ParamError( ErrorParam( e_inv_params ).extra( "[S]" ) );
      }
      else  {
         fmt->parse( *param->asString() );
         if( ! fmt->isValid() )
         {
            throw new ParseError( ErrorParam( e_param_fmt_code ) );
         }
      }
   }
}

FALCON_FUNC  Format_init ( ::Falcon::VMachine *vm )
{
   FalconObject *einst = static_cast<FalconObject*>( vm->self().asObject() );

   Format *fmt = new Format;
   einst->setUserData( fmt );

   Format_parse( vm );
}

/*#
   @method format Format
   @brief Performs desired formatting on a target item.
   @param item The item to be formatted
   @optparam dest A string where to store the formatted data.
   @return A formatted string
   @raise ParamError if a format specifier has not been set yet.
   @raise TypeError if the format specifier can't be applied the item because of
         incompatible type.

   Formats the variable as per the given format descriptor. If the class has been
   instantiated without format, and the parse() method has not been called yet,
   a ParamError is raised. If the type of the variable is incompatible with the
   format descriptor, the method returns nil; a particular format specifier allows
   to throw a TypeError in this case.

   On success, the method returns a string containing a valid formatted representation
   of the variable.

   It is possible to provide a pre-allocated string where to store the formatted
   result to improve performace and spare memory.
*/
FALCON_FUNC  Format_format ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   Format *fmt = dyncast<Format*>( einst->getFalconData() );

   Item *param = vm->param( 0 );
   Item *dest = vm->param( 1 );
   if( param == 0 || ( dest != 0 && ! dest->isString() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params ).extra( "X,[S]" ) );
   }
   else
   {
      CoreString *tgt;

      if( dest != 0 )
      {
         tgt = dest->asCoreString();
      }
      else {
         tgt = new CoreString;
      }

      if( ! fmt->format( vm, *param, *tgt ) )
         vm->retnil();
      else
         vm->retval( tgt );
   }
}


/*#
   @method toString Format
   @brief Represents this format as a string.
   @return The format specifier.

   Returns a string representation of the format instance.
*/

FALCON_FUNC  Format_toString ( ::Falcon::VMachine *vm )
{
   FalconObject *einst = dyncast< FalconObject*>( vm->self().asObject() );
   Format *fmt = dyncast< Format*>( einst->getFalconData() );
   vm->retval( new CoreString( fmt->originalFormat()) );
}

}
}

/* end of format_ext.fal */

