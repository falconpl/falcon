/*
 * hpdf_page.h
 *
 *  Created on: 03.04.2010
 *      Author: maik
 */

#ifndef FALCON_MODULE_HPDF_EXT_PAGE_H
#define FALCON_MODULE_HPDF_EXT_PAGE_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon { namespace Ext { namespace hpdf {

struct Page
{
	static void registerExtensions(Falcon::Module*);

	static FALCON_FUNC init( VMachine* );
	static FALCON_FUNC beginText( VMachine* );
	static FALCON_FUNC endText( VMachine* );
	static FALCON_FUNC showText( VMachine* );
	static FALCON_FUNC setFontAndSize( VMachine* );
	static FALCON_FUNC moveTextPos( VMachine* );
	static FALCON_FUNC getWidth( VMachine* );
	static FALCON_FUNC setWidth( VMachine* );
	static FALCON_FUNC getHeight( VMachine* );
	static FALCON_FUNC setHeight( VMachine* );
	static FALCON_FUNC getLineWidth( VMachine* );
	static FALCON_FUNC setLineWidth( VMachine* );
	static FALCON_FUNC rectangle( VMachine* );
	static FALCON_FUNC stroke( VMachine* );
	static FALCON_FUNC textWidth( VMachine* );
	static FALCON_FUNC textOut( VMachine* );
	static FALCON_FUNC lineTo( VMachine* );
	static FALCON_FUNC moveTo( VMachine* );
	static FALCON_FUNC setDash( VMachine* );
	static FALCON_FUNC setRGBStroke( VMachine* );
	static FALCON_FUNC setLineCap( VMachine* );
	static FALCON_FUNC setLineJoin( VMachine* );
	static FALCON_FUNC setRGBFill( VMachine* );
	static FALCON_FUNC fill( VMachine* );
  static FALCON_FUNC fillStroke( VMachine* );
  static FALCON_FUNC gSave( VMachine* );
  static FALCON_FUNC clip( VMachine* );
  static FALCON_FUNC setTextLeading( VMachine* );
  static FALCON_FUNC showTextNextLine( VMachine* );
  static FALCON_FUNC gRestore( VMachine* );
  static FALCON_FUNC curveTo( VMachine* );
  static FALCON_FUNC curveTo2( VMachine* );
  static FALCON_FUNC curveTo3( VMachine* );
  static FALCON_FUNC measureText( VMachine* );
  static FALCON_FUNC getCurrentFontSize( VMachine* );
  static FALCON_FUNC getCurrentFont( VMachine* );
  static FALCON_FUNC getRGBFill( VMachine* );
  static FALCON_FUNC setTextRenderingMode( VMachine* );
  static FALCON_FUNC setTextMatrix( VMachine* );
  static FALCON_FUNC setCharSpace( VMachine* );
  static FALCON_FUNC setWordSpace( VMachine* );
  static FALCON_FUNC setSize( VMachine* );
  static FALCON_FUNC textRect( VMachine* );
  static FALCON_FUNC concat( VMachine* );
  static FALCON_FUNC setGrayStroke( VMachine* );
  static FALCON_FUNC circle( VMachine* );
  static FALCON_FUNC setGrayFill( VMachine* );
};
//  FALCON_FUNC PdfPage_setRotate( VMachine* );
//  FALCON_FUNC PdfPage_createDestinatio( VMachine* );
//  FALCON_FUNC PdfPage_create3DAnnot( VMachine* );
//  FALCON_FUNC PdfPage_createTextAnnot( VMachine* );
//  FALCON_FUNC PdfPage_createLinkAnnot( VMachine* );
//  FALCON_FUNC PdfPage_createURILinkAnnot( VMachine* );
//  FALCON_FUNC PdfPage_getGMode( VMachine* );
//  FALCON_FUNC PdfPage_getCurrentPos( VMachine* );
//  FALCON_FUNC PdfPage_getCurrentPos2( VMachine* );
//  FALCON_FUNC PdfPage_getCurrentTextPos( VMachine* );
//  FALCON_FUNC PdfPage_getCurrentTextPos2( VMachine* );
//  FALCON_FUNC PdfPage_getTransMatrix( VMachine* );
//  FALCON_FUNC PdfPage_getLineCap( VMachine* );
//  FALCON_FUNC PdfPage_getLineJoin( VMachine* );
//  FALCON_FUNC PdfPage_getMiterLimit( VMachine* );
//  FALCON_FUNC PdfPage_getDash( VMachine* );
//  FALCON_FUNC PdfPage_getFlat( VMachine* );
//  FALCON_FUNC PdfPage_getCharSpace( VMachine* );
//  FALCON_FUNC PdfPage_getWordSpace( VMachine* );
//  FALCON_FUNC PdfPage_getHorizontalScalling( VMachine* );
//  FALCON_FUNC PdfPage_getTextLeading( VMachine* );
//  FALCON_FUNC PdfPage_getTextRenderingMode( VMachine* );
//  FALCON_FUNC PdfPage_getTextRaise( VMachine* );
//  FALCON_FUNC PdfPage_getTextRise( VMachine* );
//  FALCON_FUNC PdfPage_getRGBStroke( VMachine* );
//  FALCON_FUNC PdfPage_getCMYKFill( VMachine* );
//  FALCON_FUNC PdfPage_getCMYKStroke( VMachine* );
//  FALCON_FUNC PdfPage_getGrayFill ( VMachine* );
//  FALCON_FUNC PdfPage_getGrayStroke( VMachine* );
//  FALCON_FUNC PdfPage_getStrokingColorSpace( VMachine* );
//  FALCON_FUNC PdfPage_getFillingColorSpace( VMachine* );
//  FALCON_FUNC PdfPage_getGStateDepth( VMachine* );
//  FALCON_FUNC PdfPage_setMiterLimit( VMachine* );
//  FALCON_FUNC PdfPage_setFlat( VMachine* );
//  FALCON_FUNC PdfPage_setExtGState( VMachine* );
//  FALCON_FUNC PdfPage_closePath( VMachine* );
//  FALCON_FUNC PdfPage_closePathStroke( VMachine* );
//  FALCON_FUNC PdfPage_eofill( VMachine* );
//  FALCON_FUNC PdfPage_eofillStroke( VMachine* );
//  FALCON_FUNC PdfPage_closePathFillStroke( VMachine* );
//  FALCON_FUNC PdfPage_closePathEofillStroke ( VMachine* );
//  FALCON_FUNC PdfPage_endPath( VMachine* );
//  FALCON_FUNC PdfPage_eoclip ( VMachine* );

//  FALCON_FUNC PdfPage_setHorizontalScalling( VMachine* );

//  FALCON_FUNC PdfPage_setTextRise( VMachine* );
//  FALCON_FUNC PdfPage_setTextRaise( VMachine* );
//  FALCON_FUNC PdfPage_moveTextPos2( VMachine* );
//  FALCON_FUNC PdfPage_getTextMatrix( VMachine* );
//  FALCON_FUNC PdfPage_moveToNextLine( VMachine* );

//  FALCON_FUNC PdfPage_showTextNextLineEx( VMachine* );
//  FALCON_FUNC PdfPage_setCMYKFill( VMachine* );
//  FALCON_FUNC PdfPage_setCMYKStroke( VMachine* );
//  FALCON_FUNC PdfPage_executeXObject( VMachine* );
//  FALCON_FUNC PdfPage_drawImage( VMachine* );
//  FALCON_FUNC PdfPage_ellipse( VMachine* );
//  FALCON_FUNC PdfPage_arc( VMachine* );
//  FALCON_FUNC PdfPage_setSlideShow( VMachine* );

}}} // Falcon::Ext::hpdf

#endif // FALCON_MODULE_HPDF_EXT_PAGE_H
