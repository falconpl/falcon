/*
   FALCON - The Falcon Programming Language
   FILE: traceback.h

   Structure holding representation information of a point in code.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 06 Mar 2014 18:14:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_TRACEBACK_H
#define FALCON_TRACEBACK_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/tracestep.h>
#include <falcon/enumerator.h>

namespace Falcon {

/**
 * Structure holding representation information of a point in code.
 *
 * Used in errors and debugger.
 */

class FALCON_DYN_CLASS TraceBack
{
public:
   TraceBack();
   ~TraceBack();

   void add(TraceStep* ts);

   String toString( bool bAddPath = false, bool bAddParams = false) const { String temp; return toString( temp, bAddPath, bAddParams ); }
   String &toString( String &target, bool bAddPath = false, bool bAddParams = false ) const;

   length_t size() const;
   TraceStep* at(length_t pos) const;

   /** Enumerator for trace steps.
    @see enumerateSteps
   */
   typedef Enumerator<TraceStep> StepEnumerator;

   /** Enumerate the traceback steps.
    \param rator A StepEnumerator that is called back with each step in turn.
    */
   void enumerateSteps( StepEnumerator &rator ) const;

private:
   class Private;
   Private* _p;
};

}

#endif

/* end of traceback.h */
