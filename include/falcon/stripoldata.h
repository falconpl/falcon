/*
   FALCON - The Falcon Programming Language.
   FILE: stripoldata.h

   Pre-compiled data for string interpolation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 01 Feb 2013 11:06:19 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_STRIPOLDATA_H_
#define _FALCON_STRIPOLDATA_H_

#include <falcon/setup.h>
#include <falcon/genericdata.h>
#include <falcon/pstep.h>

namespace Falcon
{

class TreeStep;
class Item;
class VMContext;
class Symbol;
class Format;

/** Data for string interpolation.

\note The class is fully interlocked and MT ready.
 */
class FALCON_DYN_CLASS StrIPolData: public GenericData
{
public:
   StrIPolData();
   StrIPolData(const StrIPolData& other);
   virtual ~StrIPolData();
   
   typedef enum
   {
      e_pr_fail,
      e_pr_ok,
      e_pr_noneed
   }
   t_parse_result;

   /**
    * Parses the given string.
    * \param source The string to be parsed.
    * \param failPos Position at which the parsing failed, on failure.
    * \return A result indicating whether the opration failed, succeeded or was
    * aborted due to the fact that interpolation isn't necessary.
    *
    * This method parses a string like "... $... $(elem:...) $({expr}:...)".
    *
    * Can return:
    * - e_pr_fail: $() format parsing failed.
    * - e_pr_ok: operation complete.
    * - e_pr_noneed: The source string doesn't contain any $ sequence, and
    *    doesn't need to be interpolated.
    */
   t_parse_result parse( const String& source, int& failPos );

   virtual void gcMark( uint32 value );
   virtual bool gcCheck( uint32 value );
   
   virtual StrIPolData* clone() const;
   virtual void describe( String& target ) const;
   
   class FALCON_DYN_CLASS Slice {
   public:
      typedef enum {
         e_t_static,
         e_t_symbol,
         e_t_expr
      }
      t_type;

      t_type m_type;
      String m_def;
      Format* m_format;

      TreeStep *m_compiled;
      const Symbol* m_symbol;

      Slice( t_type t, const String& def, Format* format = 0, TreeStep* comp = 0 );
      Slice( const Slice& other );
      ~Slice();
   };

   /**
    * Adds a new slice to the interpolation data.
    * \param slice the slice to be added.
    * \return Count of slices up to date.
    */
   uint32 addSlice( Slice* slice );

   /**
    * Removes the slice at given postion.
    * \param pos The position at which the slice should be removed.
    */
   void delSlice( uint32 pos );

   /**
    * Gets the slice at the nth position.
    * \param pos The position at which the slice is found.
    * \return 0 if out of range or a valid slice.
    */
   Slice* getSlice( uint32 pos ) const;

   /**
    * Returns the count of the slices in the string.
    */
   uint32 sliceCount() const;

   /**
    * Returns the count of dynamic slices in the string.
    */
   uint32 dynSliceCount() const { return m_dynCount; }

   /** Reads the strings from this sequence of items and mounts them.
    \param Item a sequence of Items containing Strings long sliceCount() elements.
    \return 0 if some of the item is not a string, a complete new'd string otherwise.
    */
   String* mount( Item* data ) const;

   /** Pushes the data and PSteps required to get a string out of the nth slice definition.
    * \param ctx The context where to push data and steps.
    * \param nth The dynamic slice "$..." number in the source string.
    * \return true if the step can be prepared,
    *         false if nth is pointing to a static slice or is out of range.
    */
   bool prepareStep( VMContext* ctx, uint32 nth );

   void line( int32 l ) {m_line = l;}
   int32 line() const { return m_line; }

private:
   class Private;
   Private* _p;

   uint32 m_mark;
   //String m_source;
   uint32 m_dynCount;
   int32 m_line;

   class FALCON_DYN_CLASS PStepExprComp: public PStep
   {
   public:
      PStepExprComp(StrIPolData* ipd): m_owner(ipd) {apply = apply_;}
      virtual ~PStepExprComp() {}
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& desc , int ) const
      {
         desc = "StrIPolData::PStepExprComp";
      }
   private:
      StrIPolData* m_owner;
   };

   PStepExprComp m_pStepExprComp;
};

}

#endif	/* _FALCON_STRIPOLDATA_H_ */

/* end of stripoldata.h */
