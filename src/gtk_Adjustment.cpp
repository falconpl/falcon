/**
 *  \file gtk_Adjustment.cpp
 */

#include "gtk_Adjustment.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Adjustment::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Adjustment = mod->addClass( "GtkAdjustment", &Adjustment::init );

    c_Adjustment->setWKS( true );
    c_Adjustment->getClassDef()->factory( &Adjustment::factory );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkObject" ) );
    c_Adjustment->getClassDef()->addInheritance( in );

    Gtk::MethodTab methods[] =
    {
    { "signal_changed",     &Adjustment::signal_changed },
    { "signal_value_changed",&Adjustment::signal_value_changed },
    { "get_value",          &Adjustment::get_value },
    { "set_value",          &Adjustment::set_value },
    { "clamp_page",         &Adjustment::clamp_page },
    { "changed",            &Adjustment::changed },
    { "value_changed",      &Adjustment::value_changed },
#if GTK_CHECK_VERSION( 2, 14, 0 )
    { "configure",          &Adjustment::configure },
    { "get_lower",          &Adjustment::get_lower },
    { "get_page_increment", &Adjustment::get_page_increment },
    { "get_page_size",      &Adjustment::get_page_size },
    { "get_step_increment", &Adjustment::get_step_increment },
    { "get_upper",          &Adjustment::get_upper },
    { "set_lower",          &Adjustment::set_lower },
    { "set_page_increment", &Adjustment::set_page_increment },
    { "set_page_size",      &Adjustment::set_page_size },
    { "set_step_increment", &Adjustment::set_step_increment },
    { "set_upper",          &Adjustment::set_upper },
#endif // GTK_CHECK_VERSION( 2, 14, 0 )
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Adjustment, meth->name, meth->cb );
}


Adjustment::Adjustment( const Falcon::CoreClass* gen, const GtkAdjustment* adj )
    :
    Gtk::CoreGObject( gen, (GObject*) adj )
{}


Falcon::CoreObject* Adjustment::factory( const Falcon::CoreClass* gen, void* adj, bool )
{
    return new Adjustment( gen, (GtkAdjustment*) adj );
}


/*#
    @class GtkAdjustment
    @brief A GtkObject representing an adjustable bounded value
    @optparam value the current value.
    @optparam lower the minimum value.
    @optparam upper the maximum value.
    @optparam step_increment the increment to use to make minor changes to the value. In a GtkScrollbar this increment is used when the mouse is clicked on the arrows at the top and bottom of the scrollbar, to scroll by a small amount.
    @optparam page_increment the increment to use to make major changes to the value. In a GtkScrollbar this increment is used when the mouse is clicked in the trough, to scroll by a large amount.
    @optparam page_size the page size. In a GtkScrollbar this is the size of the area which is currently visible.

    The GtkAdjustment object represents a value which has an associated lower and
    upper bound, together with step and page increments, and a page size.
    It is used within several GTK+ widgets, including GtkSpinButton, GtkViewport,
    and GtkRange (which is a base class for GtkHScrollbar, GtkVScrollbar, GtkHScale,
    and GtkVScale).

    The GtkAdjustment object does not update the value itself. Instead it is left
    up to the owner of the GtkAdjustment to control the value.

    The owner of the GtkAdjustment typically calls the gtk_adjustment_value_changed()
    and gtk_adjustment_changed() functions after changing the value and its bounds.
    This results in the emission of the "value_changed" or "changed" signal respectively.
 */
FALCON_FUNC Adjustment::init( VMARG )
{
    Gtk::ArgCheck0 args( vm, "[N],[N],[N],[N],[N],[N]" );

    gdouble value = args.getNumeric( 0, false );
    gdouble lower = args.getNumeric( 1, false );
    gdouble upper = args.getNumeric( 2, false );
    gdouble step_incr = args.getNumeric( 3, false );
    gdouble page_incr = args.getNumeric( 4, false );
    gdouble page_sz = args.getNumeric( 5, false );

    MYSELF;
    self->setGObject( (GObject*) gtk_adjustment_new(
            value, lower, upper, step_incr, page_incr, page_sz ) );
}


/*#
    @method signal_changed GtkAdjustment
    @brief Emitted when one or more of the GtkAdjustment fields have been changed, other than the value field.
 */
FALCON_FUNC Adjustment::signal_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "changed", (void*) &Adjustment::on_changed, vm );
}


void Adjustment::on_changed( GtkAdjustment* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "changed", "on_changed", (VMachine*)_vm );
}


/*#
    @method signal_value_changed GtkAdjustment
    @brief Emitted when the GtkAdjustment value field has been changed.
 */
FALCON_FUNC Adjustment::signal_value_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "value_changed", (void*) &Adjustment::on_value_changed, vm );
}


void Adjustment::on_value_changed( GtkAdjustment* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "value_changed", "on_value_changed", (VMachine*)_vm );
}


/*#
    @method get_value GtkAdjustment
    @brief Gets the current value of the adjustment.
    @return The current value of the adjustment.
 */
FALCON_FUNC Adjustment::get_value( VMARG )
{
    NO_ARGS
    vm->retval( gtk_adjustment_get_value( GET_ADJUSTMENT( vm->self() ) ) );
}


/*#
    @method set_value GtkAdjustment
    @brief Sets the GtkAdjustment value.
    @param value the new value
    The value is clamped to lie between adjustment->lower and adjustment->upper.

    Note that for adjustments which are used in a GtkScrollbar, the effective range
    of allowed values goes from adjustment->lower to adjustment->upper - adjustment->page_size.
 */
FALCON_FUNC Adjustment::set_value( VMARG )
{
    Item* i_val = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_val || !i_val->isOrdinal() )
        throw_inv_params( "N" );
#endif
    gtk_adjustment_set_value( GET_ADJUSTMENT( vm->self() ),
                              i_val->forceNumeric() );
}


/*#
    @method clamp_page GtkAdjustment
    @brief Updates the GtkAdjustment value to ensure that the range between lower and upper is in the current page (i.e. between value and value + page_size).
    @param lower the lower value
    @param upper the upper value

    If the range is larger than the page size, then only the start of it will be
    in the current page. A "changed" signal will be emitted if the value is changed.
 */
FALCON_FUNC Adjustment::clamp_page( VMARG )
{
    Item* i_low = vm->param( 0 );
    Item* i_upp = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_low || !i_low->isOrdinal()
        || !i_upp || !i_upp->isOrdinal() )
        throw_inv_params( "N,N" );
#endif
    gtk_adjustment_clamp_page( GET_ADJUSTMENT( vm->self() ),
                               i_low->forceNumeric(), i_upp->forceNumeric() );
}


/*#
    @method changed GtkAdjustment
    @brief Emits a "changed" signal from the GtkAdjustment.

    This is typically called by the owner of the GtkAdjustment after it has changed
    any of the GtkAdjustment fields other than the value.
 */
FALCON_FUNC Adjustment::changed( VMARG )
{
    NO_ARGS
    gtk_adjustment_changed( GET_ADJUSTMENT( vm->self() ) );
}


/*#
    @method value_changed GtkAdjustment
    @brief Emits a "value_changed" signal from the GtkAdjustment.

    This is typically called by the owner of the GtkAdjustment after it has changed
    the GtkAdjustment value field.
 */
FALCON_FUNC Adjustment::value_changed( VMARG )
{
    NO_ARGS
    gtk_adjustment_value_changed( GET_ADJUSTMENT( vm->self() ) );
}


#if GTK_CHECK_VERSION( 2, 14, 0 )
/*#
    @method configure GtkAdjustment
    @brief Sets all properties of the adjustment at once.
    @param value the new value.
    @param lower the new minimum value.
    @param upper the new maximum value.
    @param step_increment the step increment.
    @param page_increment the page increment.
    @param page_size the page size.

    Use this function to avoid multiple emissions of the "changed" signal.
    See gtk_adjustment_set_lower() for an alternative way of compressing multiple
    emissions of "changed" into one.
 */
FALCON_FUNC Adjustment::configure( VMARG )
{
    Gtk::ArgCheck0 args( vm, "N,N,N,N,N,N" );

    gdouble value = args.getNumeric( 0 );
    gdouble lower = args.getNumeric( 1 );
    gdouble upper = args.getNumeric( 2 );
    gdouble step_incr = args.getNumeric( 3 );
    gdouble page_incr = args.getNumeric( 4 );
    gdouble page_sz = args.getNumeric( 5 );

    gtk_adjustment_configure( GET_ADJUSTMENT( vm->self() ),
            value, lower, upper, step_incr, page_incr, page_sz );
}


/*#
    @method get_lower GtkAdjustment
    @brief Retrieves the minimum value of the adjustment.
    @return The current minimum value of the adjustment.
 */
FALCON_FUNC Adjustment::get_lower( VMARG )
{
    NO_ARGS
    vm->retval( gtk_adjustment_get_lower( GET_ADJUSTMENT( vm->self() ) ) );
}


/*#
    @method get_page_increment GtkAdjustment
    @brief Retrieves the page increment of the adjustment.
    @return The current page increment of the adjustment.
 */
FALCON_FUNC Adjustment::get_page_increment( VMARG )
{
    NO_ARGS
    vm->retval( gtk_adjustment_get_page_increment( GET_ADJUSTMENT( vm->self() ) ) );
}


/*#
    @method get_page_size GtkAdjustment
    @brief Retrieves the page size of the adjustment.
    @return The current page size of the adjustment.
 */
FALCON_FUNC Adjustment::get_page_size( VMARG )
{
    NO_ARGS
    vm->retval( gtk_adjustment_get_page_size( GET_ADJUSTMENT( vm->self() ) ) );
}


/*#
    @method get_step_increment GtkAdjustment
    @brief Retrieves the step increment of the adjustment.
    @return The current step increment of the adjustment.
 */
FALCON_FUNC Adjustment::get_step_increment( VMARG )
{
    NO_ARGS
    vm->retval( gtk_adjustment_get_step_increment( GET_ADJUSTMENT( vm->self() ) ) );
}


/*#
    @method get_upper GtkAdjustment
    @brief Retrieves the maximum value of the adjustment.
    @return The current maximum value of the adjustment.
 */
FALCON_FUNC Adjustment::get_upper( VMARG )
{
    NO_ARGS
    vm->retval( gtk_adjustment_get_upper( GET_ADJUSTMENT( vm->self() ) ) );
}


/*#
    @method set_lower GtkAdjustment
    @brief Sets the minimum value of the adjustment.
    @param lower the new minimum value

    When setting multiple adjustment properties via their individual setters, multiple
    "changed" signals will be emitted. However, since the emission of the "changed"
    signal is tied to the emission of the "GObject::notify" signals of the changed
    properties, it's possible to compress the "changed" signals into one by calling
    g_object_freeze_notify() and g_object_thaw_notify() around the calls to the
    individual setters.

    Alternatively, using a single g_object_set() for all the properties to change,
    or using gtk_adjustment_configure() has the same effect of compressing "changed"
    emissions.
 */
FALCON_FUNC Adjustment::set_lower( VMARG )
{
    Item* i_low = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_low || !i_low->isOrdinal() )
        throw_inv_params( "N" );
#endif
    gtk_adjustment_set_lower( GET_ADJUSTMENT( vm->self() ),
                              i_low->forceNumeric() );
}


/*#
    @method set_page_increment GtkAdjustment
    @brief Sets the page increment of the adjustment.
    @param page_increment the new page increment

    See gtk_adjustment_set_lower() about how to compress multiple emissions of the
    "changed" signal when setting multiple adjustment properties.
 */
FALCON_FUNC Adjustment::set_page_increment( VMARG )
{
    Item* i_incr = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_incr || !i_incr->isOrdinal() )
        throw_inv_params( "N" );
#endif
    gtk_adjustment_set_page_increment( GET_ADJUSTMENT( vm->self() ),
                                       i_incr->forceNumeric() );
}


/*#
    @method set_page_size GtkAdjustment
    @brief Sets the page size of the adjustment.
    @param page_size the new page size

    See gtk_adjustment_set_lower() about how to compress multiple emissions of
    the "changed" signal when setting multiple adjustment properties.
 */
FALCON_FUNC Adjustment::set_page_size( VMARG )
{
    Item* i_sz = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_sz || !i_sz->isOrdinal() )
        throw_inv_params( "N" );
#endif
    gtk_adjustment_set_page_size( GET_ADJUSTMENT( vm->self() ),
                                  i_sz->forceNumeric() );
}


/*#
    @method set_step_increment GtkAdjustment
    @brief Sets the step increment of the adjustment.
    @param step_increment the new step increment

    See gtk_adjustment_set_lower() about how to compress multiple emissions of the
    "changed" signal when setting multiple adjustment properties.
 */
FALCON_FUNC Adjustment::set_step_increment( VMARG )
{
    Item* i_step = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_step || !i_step->isOrdinal() )
        throw_inv_params( "N" );
#endif
    gtk_adjustment_set_step_increment( GET_ADJUSTMENT( vm->self() ),
                                       i_step->forceNumeric() );
}


/*#
    @method set_upper GtkAdjustment
    @brief Sets the maximum value of the adjustment.

    Note that values will be restricted by upper - page-size if the page-size property
    is nonzero.

    See gtk_adjustment_set_lower() about how to compress multiple emissions of the
    "changed" signal when setting multiple adjustment properties.
 */
FALCON_FUNC Adjustment::set_upper( VMARG )
{
    Item* i_upp = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_upp || !i_upp->isOrdinal() )
        throw_inv_params( "N" );
#endif
    gtk_adjustment_set_upper( GET_ADJUSTMENT( vm->self() ),
                              i_upp->forceNumeric() );
}
#endif // GTK_CHECK_VERSION( 2, 14, 0 )

} // Gtk
} // Falcon
