/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi.h
 *
 * Short description
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Tue, 21 Jan 2014 15:11:56 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef _FALCON_DBI_H_
#define _FALCON_DBI_H_

#include <falcon/module.h>
#include <falcon/pstep.h>

#include "dbi_classhandle.h"
#include "dbi_classrecordset.h"
#include "dbi_classstatement.h"

namespace Falcon {

class DBIModule: public Module
{
public:
   DBIModule();
   virtual ~DBIModule();

   DBI::ClassRecordset* recordsetClass() const { return m_recordsetClass; }
   DBI::ClassHandle* handleClass() const { return m_handleClass; }
   DBI::ClassStatement* statementClass() const { return m_statementClass; }

   PStep* m_stepCatchSubmoduleLoadError;
   PStep* m_stepAfterSubmoduleLoad;

private:
   DBI::ClassRecordset* m_recordsetClass;
   DBI::ClassHandle* m_handleClass;
   DBI::ClassStatement* m_statementClass;

};

}

#endif

/* end of dbi.h */
