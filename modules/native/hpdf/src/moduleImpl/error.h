/*
 * errorhandler.h
 *
 *  Created on: 06.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_HPDF_MODIMPL_ERROR_H_
#define FALCON_MODULE_HPDF_MODIMPL_ERROR_H_


#include <falcon/error.h>
#include <hpdf.h>

#define FALCON_HPDF_ERROR_BASE 10100

namespace Falcon { namespace Mod { namespace hpdf {


void storeVMStringID(int hdf_id, int vmStringID);
int getVMStringID(int hdf_id);
void error_handler( HPDF_STATUS errorNo, HPDF_STATUS detailNo, void* user_data );

/** Class to indentify HPDF low level errors.
 * HPDF C library errors are represented to the falcon engine by instances of
 * this class */
class FALCON_DYN_CLASS Error : public Falcon::Error
{
public:
  Error():
    Falcon::Error( "HPDFError" )
  { }

  Error( ErrorParam const& params  ):
    Falcon::Error( "HPDFError", params )
  { }
};

}}} // Falcon::Mod::hpdf

#endif /* FALCON_MODULE_HPDF_MODIMPL_ERROR_H_ */
