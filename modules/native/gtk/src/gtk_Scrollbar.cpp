/**
 *  \file gtk_Scrollbar.cpp
 */

#include "gtk_Scrollbar.hpp"

#include "gtk_Adjustment.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Scrollbar::modInit( Falcon::Module* mod )
{
#if GTK_CHECK_VERSION( 3, 0, 0 )
    Falcon::Symbol* c_Scrollbar = mod->addClass( "GtkScrollbar", &Scrollbar::init );
#else
    Falcon::Symbol* c_Scrollbar = mod->addClass( "GtkScrollbar", &Gtk::abstract_init );
#endif

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkRange" ) );
    c_Scrollbar->getClassDef()->addInheritance( in );

}


/*#
    @class GtkScrollbar
    @brief Base class for GtkHScrollbar and GtkVScrollbar
    @param orientation the scrollbar's orientation (GtkOrientation).
    @param adjustment the GtkAdjustment to use, or NULL to create a new adjustment.

    The GtkScrollbar widget is the base class for GtkHScrollbar and GtkVScrollbar.
    It can be used in the same way as these, by setting the "orientation" property
    appropriately.

    The position of the thumb in a scrollbar is controlled by the scroll
    adjustments. See GtkAdjustment for the fields in an adjustment - for
    GtkScrollbar, the "value" field represents the position of the scrollbar,
    which must be between the "lower" field and "upper - page_size."
    The "page_size" field represents the size of the visible scrollable area.
    The "step_increment" and "page_increment" fields are used when the user
    asks to step down (using the small stepper arrows) or page down (using
    for example the PageDown key).
 */
#if GTK_CHECK_VERSION( 3, 0, 0 )
FALCON_FUNC Scrollbar::init( VMARG )
{
    Item* i_ori = vm->param( 0 );
    Item* i_adj = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ori || !i_ori->isInteger()
        || !i_adj || !( i_adj->isNil() || ( i_adj->isObject()
        && IS_DERIVED( i_adj, GtkAdjustment ) ) ) )
        throw_inv_params( "GtkOrientation,[GtkAdjustment]" );
#endif
    GtkAdjustment* adj = i_adj->isNil() ? NULL : GET_ADJUSTMENT( *i_adj );
    MYSELF;
    self->setObject( (GObject*) gtk_scrollbar_new( (GtkOrientation) i_ori->asInteger(),
                                                   adj ) );
}
#endif

} // Gtk
} // Falcon
