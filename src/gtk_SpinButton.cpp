/**
 *  \file gtk_SpinButton.cpp
 */

#include "gtk_SpinButton.hpp"

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */

void SpinButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_SpinButton = mod->addClass( "GtkSpinButton", &SpinButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkEntry" ) );
    c_SpinButton->getClassDef()->addInheritance( in );

    c_SpinButton->getClassDef()->factory( &SpinButton::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_change_value",&SpinButton::signal_change_value },
    { "signal_input",       &SpinButton::signal_input },
    { "signal_output",      &SpinButton::signal_output },
    { "signal_value_changed",&SpinButton::signal_value_changed },
    { "signal_wrapped",     &SpinButton::signal_wrapped },
    //{ "set_adjustment",     &SpinButton::set_adjustment },
    //{ "get_adjustment",     &SpinButton::get_adjustment },
    { "set_digits",         &SpinButton::set_digits },
    { "set_increments",     &SpinButton::set_increments },
    { "set_range",          &SpinButton::set_range },
    { "get_value_as_int",   &SpinButton::get_value_as_int },
    { "set_value",          &SpinButton::set_value },
    { "set_update_policy",  &SpinButton::set_update_policy },
    { "set_numeric",        &SpinButton::set_numeric },
    { "spin",               &SpinButton::spin },
    { "set_wrap",           &SpinButton::set_wrap },
    { "set_snap_to_ticks",  &SpinButton::set_snap_to_ticks },
    { "update",             &SpinButton::update },
    { "get_digits",         &SpinButton::get_digits },
    { "get_increments",     &SpinButton::get_increments },
    { "get_numeric",        &SpinButton::get_numeric },
    { "get_range",          &SpinButton::get_range },
    { "get_snap_to_ticks",  &SpinButton::get_snap_to_ticks },
    { "get_update_policy",  &SpinButton::get_update_policy },
    { "get_value",          &SpinButton::get_value },
    { "get_wrap",           &SpinButton::get_wrap },
    { NULL, NULL }
    };

    for( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_SpinButton, meth->name, meth->cb );
}


SpinButton::SpinButton( const Falcon::CoreClass* gen, const GtkSpinButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* SpinButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new SpinButton( gen, (GtkSpinButton*) btn );
}


/*#
    @class gtk.SpinButton
    @brief Retrieve an integer or floating-point number from the user
    @optparam min,max,step
    @optparam adjustment,climb_rate,digits

    A GtkSpinButton is an ideal way to allow the user to set the value of some attribute. Rather than having
    to directly type a number into a GtkEntry, GtkSpinButton allows the user to click on one of two arrows to
    increment or decrement the displayed value. A value can still be typed in, with the bonus that it can be
    checked to ensure it is in a given range.

    The main properties of a GtkSpinButton are through a GtkAdjustment. See the GtkAdjustment section for more
    details about an adjustment's properties.
*/
FALCON_FUNC SpinButton::init( VMARG )
{
    MYSELF;

    if ( self->getGObject() )
        return;

    Item* i_first = vm->param( 0 );
    Item* i_second = vm->param( 1 );
    Item* i_third = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if( !i_first || !i_first->isOrdinal() ||
        !i_second || !i_second->isOrdinal() ||
        !i_third || !i_third->isOrdinal() ) //TODO add Paramter checks for gtk_spin_button_new()
    {
        throw_inv_params( "N,N,N" );
    }
#endif

    GtkWidget* spinbutton;
    if( i_first->isOrdinal() )
    {
        spinbutton = gtk_spin_button_new_with_range( (gdouble)i_first->forceNumeric(), (gdouble)i_second->forceNumeric(), (gdouble)i_third->forceNumeric() );
    }
    else
    {
        //TODO add gtk_spin_button_new paramterchecks and types
    }

    self->setGObject( (GObject*) spinbutton );
}

/*#
    @method signal_change_value GtkSpinButton
    @brief Connect a VMSlot to the spinbutton change-value signal and return it
 */
FALCON_FUNC SpinButton::signal_change_value( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif

    CoreGObject::get_signal( "change_value", (void*) &SpinButton::on_change_value, vm );
}


void SpinButton::on_change_value( GtkSpinButton* btn, GtkScrollType type, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "change_value", "on_change_value", (VMachine*)_vm );
}

/*#
    @method signal_input GtkSpinButton
    @brief Connect a VMSlot to the spinbutton change-value signal and return it
 */
FALCON_FUNC SpinButton::signal_input( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif

    CoreGObject::get_signal( "input", (void*) &SpinButton::on_input, vm );
}


gint SpinButton::on_input( GtkSpinButton* btn, gpointer arg1, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "input", "on_input", (VMachine*)_vm );

    return (gint)((VMachine*)_vm)->regA().forceInteger();
}

/*#
    @method signal_output GtkSpinButton
    @brief Connect a VMSlot to the spinbutton output signal and return it
 */
FALCON_FUNC SpinButton::signal_output( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif

    CoreGObject::get_signal( "output", (void*) &SpinButton::on_output, vm );
}


gboolean SpinButton::on_output( GtkSpinButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "output", "on_output", (VMachine*)_vm );

    return (gboolean)((VMachine*)_vm)->regA().asBoolean();
}

/*#
    @method signal_value_changed GtkSpinButton
    @brief Connect a VMSlot to the spinbutton value_changed signal and return it
 */
FALCON_FUNC SpinButton::signal_value_changed( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif

    CoreGObject::get_signal( "value_changed", (void*) &SpinButton::on_value_changed, vm );
}


void SpinButton::on_value_changed( GtkSpinButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "value_changed", "on_value_changed", (VMachine*)_vm );
}

/*#
    @method signal_wrapped GtkSpinButton
    @brief Connect a VMSlot to the spinbutton wrapped signal and return it
 */
FALCON_FUNC SpinButton::signal_wrapped( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif

    CoreGObject::get_signal( "wrapped", (void*) &SpinButton::on_wrapped, vm );
}


void SpinButton::on_wrapped( GtkSpinButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "wrapped", "on_wrapped", (VMachine*)_vm );
}

/*#
    @method set_digits gtk.SpinButton
    @brief Set the precision to be displayed by spin_button. Up to 20 digit precision is allowed.
    @param digits
*/
FALCON_FUNC SpinButton::set_digits( VMARG )
{
    Item* i_digits = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if( !i_digits || !i_digits->isInteger() )
    {
        throw_inv_params( "N" );
    }
#endif

    MYSELF;
    GET_OBJ( self );

    gtk_spin_button_set_digits( (GtkSpinButton*)_obj, i_digits->asInteger() );
}

/*#
    @method set_increments gtk.SpinButton
    @brief Sets the step and page increments for spin_button. This affects how quickly the value changes when the spin button's arrows are activated.
    @param step increment applied for a button 1 press.
    @param page increment applied for a button 2 press.
*/
FALCON_FUNC SpinButton::set_increments( VMARG )
{
    Item* i_step = vm->param( 0 );
    Item* i_page = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if( !i_step || !i_step->isOrdinal() ||
        !i_page || !i_page->isOrdinal() )
    {
        throw_inv_params( "N,N" );
    }
#endif

    MYSELF;
    GET_OBJ( self );

    gtk_spin_button_set_increments( (GtkSpinButton*)_obj, (gdouble)i_step->forceNumeric(), (gdouble)i_page->forceNumeric() );
}

/*#
    @method set_range gtk.SpinButton
    @brief Sets the minimum and maximum allowable values for spin_button
    @param min minimum allowable value
    @param max 	 maximum allowable value
*/
FALCON_FUNC SpinButton::set_range( VMARG )
{
    Item* i_min = vm->param( 0 );
    Item* i_max = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if( !i_min || !i_min->isOrdinal() ||
        !i_max || !i_max->isOrdinal() )
    {
        throw_inv_params( "N,N" );
    }
#endif

    MYSELF;
    GET_OBJ( self );

    gtk_spin_button_set_range( (GtkSpinButton*)_obj, (gdouble)i_min->forceNumeric(), (gdouble)i_max->forceNumeric() );
}

/*#
    @method get_value_as_int gtk.SpinButton
    @brief Get the value spin_button represented as an integer.
    @return (integer) the value of spin_button
*/
FALCON_FUNC SpinButton::get_value_as_int( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif

    MYSELF;
    GET_OBJ( self );

    vm->retval( (Falcon::int64)gtk_spin_button_get_value_as_int( (GtkSpinButton*)_obj ) );
}

/*#
    @method set_value gtk.SpinButton
    @brief Set the value of spin_button.
    @param value the new value
*/
FALCON_FUNC SpinButton::set_value( VMARG )
{
    Item* i_value = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if( !i_value || !i_value->isOrdinal() )
    {
        throw_inv_params( "N" );
    }
#endif

    MYSELF;
    GET_OBJ( self );

    gtk_spin_button_set_value( (GtkSpinButton*)_obj, (gdouble)i_value->forceNumeric() );
}

/*#
    @method set_update_policy gtk.SpinButton
    @brief Sets the update behavior of a spin button. This determines whether the spin button is always updated or only when a valid value is set.
    @param policy A GtkSpinButtonUpdatePolicy value
*/
FALCON_FUNC SpinButton::set_update_policy( VMARG )
{
    Item* i_policy = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if( !i_policy || !i_policy->isInteger() )
    {
        throw_inv_params( "GtkSpinButtonUpdatePolicy" );
    }
#endif

    MYSELF;
    GET_OBJ( self );

    gtk_spin_button_set_update_policy( (GtkSpinButton*)_obj, (GtkSpinButtonUpdatePolicy)i_policy->asInteger() );
}

/*#
    @method set_numeric gtk.SpinButton
    @brief Sets the flag that determines if non-numeric text can be typed into the spin button.
    @param numeric flag indicating if only numeric entry is allowed.
*/
FALCON_FUNC SpinButton::set_numeric( VMARG )
{
    Item* i_numeric = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if( !i_numeric || !i_numeric->isBoolean() )
    {
        throw_inv_params( "B" );
    }
#endif

    MYSELF;
    GET_OBJ( self );

    gtk_spin_button_set_numeric( (GtkSpinButton*)_obj, (gboolean)i_numeric->asBoolean() );
}

/*#
    @method spin gtk.SpinButton
    @brief Increment or decrement a spin button's value in a specified direction by a specified amount.
    @param direction a GtkSpinType indicating the direction to spin.
    @param increment step increment to apply in the specified direction.
*/
FALCON_FUNC SpinButton::spin( VMARG )
{
    Item* i_direction = vm->param( 0 );
    Item* i_increment = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if( !i_direction || !i_direction->isInteger() ||
        !i_increment || !i_increment->isOrdinal() )
    {
        throw_inv_params( "GtkSpinType,N" );
    }
#endif

    MYSELF;
    GET_OBJ( self );

    gtk_spin_button_spin( (GtkSpinButton*)_obj, (GtkSpinType)i_direction->asInteger(), (gdouble)i_increment->forceNumeric() );
}

/*#
    @method set_wrap gtk.SpinButton
    @brief Sets the flag that determines if a spin button value wraps around to the opposite limit when the upper or lower limit of the range is exceeded.
    @param wrap a flag indicating if wrapping behavior is performed.
*/
FALCON_FUNC SpinButton::set_wrap( VMARG )
{
    Item* i_wrap = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if( !i_wrap || !i_wrap->isBoolean() )
    {
        throw_inv_params( "B" );
    }
#endif

    MYSELF;
    GET_OBJ( self );

    gtk_spin_button_set_wrap( (GtkSpinButton*)_obj, (gboolean)i_wrap->asBoolean() );
}

/*#
    @method set_snap_to_ticks gtk.SpinButton
    @brief Sets the policy as to whether values are corrected to the nearest step increment when a spin button is activated after providing an invalid value.
    @param snap_to_ticks a flag indicating if invalid values should be corrected.
*/
FALCON_FUNC SpinButton::set_snap_to_ticks( VMARG )
{
    Item* i_snap_to_ticks = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if( !i_snap_to_ticks || !i_snap_to_ticks->isBoolean() )
    {
        throw_inv_params( "B" );
    }
#endif

    MYSELF;
    GET_OBJ( self );

    gtk_spin_button_set_snap_to_ticks( (GtkSpinButton*)_obj, (gboolean)i_snap_to_ticks->asBoolean() );
}

/*#
    @method update gtk.SpinButton
    @brief Manually force an update of the spin button.
*/

FALCON_FUNC SpinButton::update( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif

    MYSELF;
    GET_OBJ( self );

    gtk_spin_button_update( (GtkSpinButton*)_obj );
}

/*#
    @method get_digits gtk.SpinButton
    @brief Fetches the precision of spin_button. See gtk_spin_button_set_digits().
    @return (integer) the current precision
*/
FALCON_FUNC SpinButton::get_digits( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif

    MYSELF;
    GET_OBJ( self );

    vm->retval( (Falcon::int64)gtk_spin_button_get_digits( (GtkSpinButton*)_obj ) );
}

/*#
    @method get_increments gtk.SpinButton
    @brief Gets the current step and page the increments used by spin_button. See gtk_spin_button_set_increments().
    @return (array) First element is the step increment, Second element is the page increment
*/
FALCON_FUNC SpinButton::get_increments( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif

    MYSELF;
    GET_OBJ( self );
    gdouble step, page;

    gtk_spin_button_get_increments( (GtkSpinButton*)_obj, &step, &page );
    CoreArray* result = new CoreArray( 2 );
    result->append( (Falcon::numeric)step );
    result->append( (Falcon::numeric)page );
    vm->retval( result );
}

/*#
    @method get_numeric gtk.SpinButton
    @brief Returns whether non-numeric text can be typed into the spin button. See gtk_spin_button_set_numeric().
    @return (boolean) TRUE if only numeric text can be entered
*/
FALCON_FUNC SpinButton::get_numeric( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif

    MYSELF;
    GET_OBJ( self );

    vm->retval( (bool)gtk_spin_button_get_numeric( (GtkSpinButton*)_obj ) );
}

/*#
    @method get_range gtk.SpinButton
    @briefGets the range allowed for spin_button. See gtk_spin_button_set_range().
    @return (array) First element is the min range, Second element is the max range
*/
FALCON_FUNC SpinButton::get_range( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif

    MYSELF;
    GET_OBJ( self );
    gdouble min, max;

    gtk_spin_button_get_range( (GtkSpinButton*)_obj, &min, &max );
    CoreArray* result = new CoreArray( 2 );
    result->append( (Falcon::numeric)min );
    result->append( (Falcon::numeric)max );
    vm->retval( result );
}

/*#
    @method get_snap_to_ticks gtk.SpinButton
    @brief Returns whether the values are corrected to the nearest step. See gtk_spin_button_set_snap_to_ticks().
    @return (boolean) TRUE if values are snapped to the nearest step.
*/
FALCON_FUNC SpinButton::get_snap_to_ticks( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif

    MYSELF;
    GET_OBJ( self );

    vm->retval( (bool)gtk_spin_button_get_snap_to_ticks( (GtkSpinButton*)_obj ) );
}

/*#
    @method get_update_policy gtk.SpinButton
    @brief Gets the update behavior of a spin button. See gtk_spin_button_set_update_policy().
    @return (GtkSpinButtonUpdatePolicy) the current update policy
*/
FALCON_FUNC SpinButton::get_update_policy( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif

    MYSELF;
    GET_OBJ( self );

    vm->retval( (Falcon::int64) gtk_spin_button_get_update_policy( (GtkSpinButton*)_obj ) );
}

/*#
    @method get_value gtk.SpinButton
    @brief Get the value in the spin_button.
    @return (number) the value of spin_button
*/
FALCON_FUNC SpinButton::get_value( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif

    MYSELF;
    GET_OBJ( self );

    vm->retval( (Falcon::numeric) gtk_spin_button_get_value( (GtkSpinButton*)_obj ) );
}

/*#
    @method get_wrap gtk.SpinButton
    @briefReturns whether the spin button's value wraps around to the opposite limit when the upper or lower limit of the range is exceeded. See gtk_spin_button_set_wrap().
    @return (boolean) 	 TRUE if the spin button wraps around
*/
FALCON_FUNC SpinButton::get_wrap( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif

    MYSELF;
    GET_OBJ( self );

    vm->retval( (bool)gtk_spin_button_get_wrap( (GtkSpinButton*)_obj ) );
}

} //Gtk
} //Falcon



