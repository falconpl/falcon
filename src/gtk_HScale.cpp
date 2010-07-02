/**
 *  \file gtk_HScale.cpp
 */

#include "gtk_HScale.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void HScale::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_HScale = mod->addClass( "GtkHScale", &HScale::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkScale" ) );
    c_HScale->getClassDef()->addInheritance( in );

    c_HScale->setWKS( true );
    c_HScale->getClassDef()->factory( &HScale::factory );

    mod->addClassMethod( c_HScale, "new_with_range", &HScale::new_with_range );
}


HScale::HScale( const Falcon::CoreClass* gen, const GtkHScale* scale )
    :
    Gtk::CoreGObject( gen, (GObject*) scale )
{}


Falcon::CoreObject* HScale::factory( const Falcon::CoreClass* gen, void* scale, bool )
{
    return new HScale( gen, (GtkHScale*) scale );
}


/*#
    @class GtkHScale
    @brief A horizontal slider widget for selecting a value from a range
    @param adjustment the GtkAdjustment which sets the range of the scale.

    The GtkHScale widget is used to allow the user to select a value using a
    horizontal slider.

    The position to show the current value, and the number of decimal places
    shown can be set using the parent GtkScale class's functions.
 */
FALCON_FUNC HScale::init( VMARG )
{
    Item* i_adj = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_adj || !i_adj->isObject() || !IS_DERIVED( i_adj, GtkAdjustment ) )
        throw_inv_params( "GtkAdjustment" );
#endif
    GtkAdjustment* adj = (GtkAdjustment*) COREGOBJECT( i_adj )->getObject();
    MYSELF;
    self->setObject( (GObject*) gtk_hscale_new( adj ) );
}


/*#
    @method new_with_range GtkHScale
    @brief Creates a new horizontal scale widget that lets the user input a number between min and max (including min and max) with the increment step.
    @param min minimum value
    @param max maximum value
    @param step step increment (tick size) used with keyboard shortcuts
    @return a new GtkHScale

    step must be nonzero; it's the distance the slider moves when using the arrow
    keys to adjust the scale value.

    Note that the way in which the precision is derived works best if step is a
    power of ten. If the resulting precision is not suitable for your needs,
    use gtk_scale_set_digits() to correct it.
 */
FALCON_FUNC HScale::new_with_range( VMARG )
{
    Item* i_min = vm->param( 0 );
    Item* i_max = vm->param( 1 );
    Item* i_step = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_min || !i_min->isOrdinal()
        || !i_max || !i_max->isOrdinal()
        || !i_step || !i_step->isOrdinal() )
        throw_inv_params( "O,O,O" );
#endif
    GtkWidget* wdt = gtk_hscale_new_with_range( i_min->asNumeric(),
                                                i_max->asNumeric(),
                                                i_step->asNumeric() );
    vm->retval( new Gtk::HScale( vm->findWKI( "GtkHScale" )->asClass(), (GtkHScale*) wdt ) );
}


} // Gtk
} // Falcon
