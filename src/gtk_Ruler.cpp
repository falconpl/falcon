/**
 *  \file gtk_Ruler.cpp
 */

#include "gtk_Ruler.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Ruler::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Ruler = mod->addClass( "GtkRuler", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkWidget" ) );
    c_Ruler->getClassDef()->addInheritance( in );

    Gtk::MethodTab methods[] =
    {
    { "set_metric",     &Ruler::set_metric },
    { "set_range",      &Ruler::set_range },
    { "get_metric",     &Ruler::get_metric },
    { "get_range",      &Ruler::get_range },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Ruler, meth->name, meth->cb );
}


Ruler::Ruler( const Falcon::CoreClass* gen, const GtkRuler* ruler )
    :
    Gtk::CoreGObject( gen, (GObject*) ruler )
{}


Falcon::CoreObject* Ruler::factory( const Falcon::CoreClass* gen, void* ruler, bool )
{
    return new Ruler( gen, (GtkRuler*) ruler );
}


/*#
    @class GtkRuler
    @brief Base class for horizontal or vertical rulers

    Note: This widget is considered too specialized/little-used for GTK+, and will
    in the future be moved to some other package. If your application needs this widget,
    feel free to use it, as the widget does work and is useful in some applications;
    it's just not of general interest. However, we are not accepting new features for
    the widget, and it will eventually move out of the GTK+ distribution.

    The GTKRuler widget is a base class for horizontal and vertical rulers. Rulers are
    used to show the mouse pointer's location in a window. The ruler can either be
    horizontal or vertical on the window. Within the ruler a small triangle indicates
    the location of the mouse relative to the horizontal or vertical ruler. See GtkHRuler
    to learn how to create a new horizontal ruler. See GtkVRuler to learn how to create
    a new vertical ruler.
 */


/*#
    @method set_metric GtkRuler
    @brief This calls the GTKMetricType to set the ruler to units defined.
    @param metric the unit of measurement

    Available units are GTK_PIXELS, GTK_INCHES, or GTK_CENTIMETERS. The default
    unit of measurement is GTK_PIXELS.
 */
FALCON_FUNC Ruler::set_metric( VMARG )
{
    Item* i_met = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_met || i_met->isNil() || !i_met->isInteger() )
        throw_inv_params( "GtkMetricType" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_ruler_set_metric( (GtkRuler*)_obj, (GtkMetricType) i_met->asInteger() );
}


/*#
    @method set_range GtkRuler
    @brief This sets the range of the ruler.
    @param lower the lower limit of the ruler
    @param upper the upper limit of the ruler
    @param position the mark on the ruler
    @param max_size the maximum size of the ruler used when calculating the space to leave for the text
 */
FALCON_FUNC Ruler::set_range( VMARG )
{
    Gtk::ArgCheck0 args( vm, "N,N,N,N" );
    double lower = args.getNumeric( 0 );
    double upper = args.getNumeric( 1 );
    double pos = args.getNumeric( 2 );
    double max = args.getNumeric( 3 );
    MYSELF;
    GET_OBJ( self );
    gtk_ruler_set_range( (GtkRuler*)_obj, lower, upper, pos, max );
}


/*#
    @method get_metric GtkRuler
    @brief Gets the units used for a GtkRuler.
    @return the units currently used for ruler
 */
FALCON_FUNC Ruler::get_metric( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_ruler_get_metric( (GtkRuler*)_obj ) );
}


/*#
    @method get_range GtkRuler
    @brief Retrieves values indicating the range and current position of a GtkRuler.
    @return [ lower, upper, position, max_size ]
 */
FALCON_FUNC Ruler::get_range( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    gdouble lower, upper, pos, max;
    MYSELF;
    GET_OBJ( self );
    gtk_ruler_get_range( (GtkRuler*)_obj, &lower, &upper, &pos, &max );
    CoreArray* arr = new CoreArray( 4 );
    arr->append( lower );
    arr->append( upper );
    arr->append( pos );
    arr->append( max );
    vm->retval( arr );
}


} // Gtk
} // Falcon
