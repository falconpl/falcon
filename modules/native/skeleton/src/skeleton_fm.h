/*
   @{MAIN_PRJ}@
   FILE: @{PROJECT_NAME}@_fm.h

   @{DESCRIPTION}@
   Interface extension functions
   -------------------------------------------------------------------
   Author: @{AUTHOR}@
   Begin: @{DATE}@

   -------------------------------------------------------------------
   (C) Copyright @{YEAR}@: @{COPYRIGHT}@

   @{LICENSE}@
*/


#ifndef _@{PROJECT_NAME}@_FM_H
#define _@{PROJECT_NAME}@_FM_H

#include <falcon/module.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

class Module@{MODULE_NAME}@: public Module
{
public:
   Module@{MODULE_NAME}@();
   virtual ~Module@{MODULE_NAME}@();

   // A PStep that is used by some function in our code.
   const PStep* stepIterate() const { return m_stepIterate; }

public:

   const PStep* m_stepIterate;
};

}} // namespace Falcon::Ext

#endif

/* end of @{PROJECT_NAME}@_fm.h */
