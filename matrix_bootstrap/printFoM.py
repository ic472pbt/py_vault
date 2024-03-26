#!/opt/intel/oneapi/intelpython/latest/bin/python

import sys
import numpy as np

# Common bias - 6th-to-10th bins
# Number of arguments
num_args = len(sys.argv)

# Get help for parameters order
if (num_args == 2 and sys.argv[1] == '-h'):
    print(' arguments : filename, pess/opt, probe, num_bias_sp, num_bias_ph, curvature, gamma, spec alpha, spec and photo alpha')
else:
    # Input parameters
    script = sys.argv[0]
    # Fisher matrix file
    filename = sys.argv[1]
    # Pess/Opt
    case = sys.argv[2]
    # peculiar velocities
    sigp = 'sig_p'
    sigv = 'sig_v'
    # probe
    probe = sys.argv[3]
    # Number of bias spectro
    num_bias_sp = int(sys.argv[4])
    # Number of bias photo
    num_bias_ph = int(sys.argv[5])
    # Curvature : Flat (F) or Non-Flat (NF)
    curvature = sys.argv[6]
    # Growth index
    gamma = sys.argv[7]
    # Peculiar velocities index
    index_sig = np.array([9, 10])
    # Maximum possible number of cosmo parameters
    max_num_params = 9

    # Check PESS/OPT parameter
    if (case != 'PESS' and case != 'OPT'):
        print('')
        print('Bad pessimistic/optimistic parameter : PESS or OPT')
        sys.exit()
    # Check allowed values for curvature and gamma
    if (curvature != 'F' and curvature != 'NF'):
        print('')
        print('Bad curvature parameter : N or NF')
        sys.exit()
    if (gamma != 'Y' and gamma != 'N'):
        print('')
        print('Bad gamma parameter : Y or N')
        sys.exit()

    # First : load Fisher matrix
    C_in = np.genfromtxt(filename)
    # Get size of C_in
    size = len(C_in)
    # Test square matrix : reshape matrix into size x size
    C_in = np.reshape(C_in, (size, size))

    # Probe : GCsp_NO_XC : GCsp + GCph + WL
    if (probe == 'GCsp_NO_XC'):

        # Defaut parameters : photo convention adopted from GCsp
        terms_fiducial = np.array(
            ['wm', 'wde', 'wb', 'w0', 'wa', 'h', 'ns', 's8', 'gamma'])
        if (case == 'PESS'):
            terms_fiducial = np.append(terms_fiducial, ['sig_p', 'sig_v'])
        # Adding specro (bias,P_shot) with num_bias_sp input
        for i in range(1, num_bias_sp+1):
            terms_fiducial = np.append(terms_fiducial, 'bs'+str(i))
        for i in range(1, num_bias_sp+1):
            terms_fiducial = np.append(terms_fiducial, 'ps'+str(i))
        # Add 3 IA
        terms_fiducial = np.append(terms_fiducial, ['A_IA', 'n_IA', 'B_IA'])
        # Adding photo bias with num_bias_ph input
        for i in range(1, num_bias_ph+1):
          # Common bias) - 6th-to-10th bins
          if (num_args > 9) and (sys.argv[9] == '2'):
            if (i > 5 and i < 11):
              terms_fiducial = np.append(terms_fiducial, 'alpha'+str(i))
            else:
              terms_fiducial = np.append(terms_fiducial, 'bp'+str(i))
          else:
            terms_fiducial = np.append(terms_fiducial, 'bp'+str(i))

        # Remove elements : np.delete
        if (curvature == 'F'):
            terms = np.delete(terms_fiducial, 1, axis=0)
            if (gamma == 'N'):
                terms = np.delete(terms_fiducial, [1, 8], axis=0)
        elif (curvature == 'NF'):
            terms = np.copy(terms_fiducial)
            if (gamma == 'N'):
                terms = np.delete(terms_fiducial, 8, axis=0)
        # Test on consistent number of parameters
        number_params = len(terms)
        if (curvature == 'F'):
            if (gamma == 'N'):
                # 7 cosmo parameters (no-gamma + F) = 7, 3 IA, 10 bias => should have 20 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error1 : different parameters not consistent in size')
                    sys.exit()
            elif (gamma == 'Y'):
                # 8 cosmo parameters (gamma + F) = 8, 3 IA, 10 bias => should have 21 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error2 : different parameters not consistent in size')
                    sys.exit()
        if (curvature == 'NF'):
            if (gamma == 'N'):
                # 8 cosmo parameters (no-gamma + NF) = 8 , 3 IA, 10 bias => should have 21 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error3 : different parameters not consistent in size')
                    sys.exit()
            elif (gamma == 'Y'):
                # 9 cosmo parameters (gamma + NF) = 9, 3 IA, 10 bias => should have 22 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error4 : different parameters not consistent in size')
                    sys.exit()

    # Probe : GCsp-XC : GCsp + GCph + WL + XC
    if (probe == 'GCsp_XC'):

        # Defaut parameters : photo convention adopted from GCsp
        terms_fiducial = np.array(
            ['wm', 'wde', 'wb', 'w0', 'wa', 'h', 'ns', 's8', 'gamma'])
        if (case == 'PESS'):
            terms_fiducial = np.append(terms_fiducial, ['sig_p', 'sig_v'])
        # 2 Possiblities : 
        # 2.1) Adding specro (bias,P_shot) with num_bias_sp input
        # 2.2) print alpha parameter (alpha = b_spec/b_ph)  instead of bias
        #print('num_args =', num_args)
        #print('sys.argv[8] =', sys.argv[8])
        if (num_args > 8) and  (sys.argv[8] == '1'):
          for i in range(1, num_bias_sp+1):
            terms_fiducial = np.append(terms_fiducial, 'alpha'+str(i))
        else:
          for i in range(1, num_bias_sp+1):
            terms_fiducial = np.append(terms_fiducial, 'bs'+str(i))
        for i in range(1, num_bias_sp+1):
          terms_fiducial = np.append(terms_fiducial, 'ps'+str(i))

        # Add 3 IA
        terms_fiducial = np.append(terms_fiducial, ['A_IA', 'n_IA', 'B_IA'])
        # Adding photo bias with num_bias_ph input
        # Common bias - 6th-to-10th bins
        if (num_args > 9) and (sys.argv[9] == '2'):
          for i in range(1, num_bias_ph+1):
              if (i > 5 and i < 11):
                terms_fiducial = np.append(terms_fiducial, 'alpha'+str(i))
              else:
                terms_fiducial = np.append(terms_fiducial, 'bp'+str(i))
        else:
          for i in range(1, num_bias_ph+1):
            terms_fiducial = np.append(terms_fiducial, 'bp'+str(i))

        # Remove elements : np.delete
        if (curvature == 'F'):
            terms = np.delete(terms_fiducial, 1, axis=0)
            if (gamma == 'N'):
                terms = np.delete(terms_fiducial, [1, 8], axis=0)
        elif (curvature == 'NF'):
            terms = np.copy(terms_fiducial)
            if (gamma == 'N'):
                terms = np.delete(terms_fiducial, 8, axis=0)
        # Test on consistent number of parameters
        number_params = len(terms)
        if (curvature == 'F'):
            if (gamma == 'N'):
                # 7 cosmo parameters (no-gamma + F) = 7, 3 IA, 10 bias => should have 20 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error5 : different parameters not consistent in size')
                    sys.exit()
            elif (gamma == 'Y'):
                # 8 cosmo parameters (gamma + F) = 8, 3 IA, 10 bias => should have 21 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error6 : different parameters not consistent in size')
                    sys.exit()
        if (curvature == 'NF'):
            if (gamma == 'N'):
                # 8 cosmo parameters (no-gamma + NF) = 8 , 3 IA, 10 bias => should have 21 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error7 : different parameters not consistent in size')
                    sys.exit()
            elif (gamma == 'Y'):
                # 9 cosmo parameters (gamma + NF) = 9, 3 IA, 10 bias => should have 22 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error8 : different parameters not consistent in size')
                    sys.exit()
    # Probe : XC
    if (probe == 'XC'):
        terms_fiducial = ['wm', 'wde', 'wb', 'w0', 'wa',
                          'h', 'ns', 's8', 'gamma', 'A_IA', 'n_IA', 'B_IA']
        # Adding bias with num_bias input
        for i in range(1, num_bias_ph+1):
          # Common bias - 6th-to-10th bins
          if (num_args > 9) and (sys.argv[9] == '2'):
              if (i > 5 and i < 11):
                terms_fiducial = np.append(terms_fiducial, 'alpha'+str(i))
              else:
                terms_fiducial = np.append(terms_fiducial, 'bp'+str(i))
          else:
            terms_fiducial = np.append(terms_fiducial, 'bp'+str(i))
        if (curvature == 'F'):
            terms = np.delete(terms_fiducial, 1, axis=0)
            if (gamma == 'N'):
                terms = np.delete(terms_fiducial, [1, 8], axis=0)
            elif (gamma == 'Y'):
                #terms = np.copy(terms_fiducial)
                pass
        elif (curvature == 'NF'):
            if (gamma == 'N'):
                terms = np.delete(terms_fiducial, 8, axis=0)
            elif (gamma == 'Y'):
                terms = np.copy(terms_fiducial)
        # Test on consistent number of parameters
        number_params = len(terms)
        if (curvature == 'F'):
            if (gamma == 'N'):
                # 7 cosmo parameters (no-gamma + F) = 7, 3 IA, 10 bias => should have 20 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error9 : different parameters not consistent in size')
                    sys.exit()
            elif (gamma == 'Y'):
                # 8 cosmo parameters (gamma + F) = 8, 3 IA, 10 bias => should have 21 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error10 : different parameters not consistent in size')
                    sys.exit()
        if (curvature == 'NF'):
            if (gamma == 'N'):
                # 8 cosmo parameters (no-gamma + NF) = 8 , 3 IA, 10 bias => should have 21 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error11 : different parameters not consistent in size')
                    sys.exit()
            elif (gamma == 'Y'):
                # 9 cosmo parameters (gamma + NF) = 9, 3 IA, 10 bias => should have 22 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error12 : different parameters not consistent in size')
                    sys.exit()
    # Probe : GCph
    if (probe == 'GCph'):
        terms_fiducial = ['wm', 'wde', 'wb',
                          'w0', 'wa', 'h', 'ns', 's8', 'gamma']
        # Adding bias with num_bias_ph input
        for i in range(1, num_bias_ph+1):
          # Common bias - 6th-to-10th bins
          if (num_args > 9) and (sys.argv[9] == '2'):
              if (i > 5 and i < 11):
                terms_fiducial = np.append(terms_fiducial, 'alpha'+str(i))
              else:
                terms_fiducial = np.append(terms_fiducial, 'bp'+str(i))
          else:
            terms_fiducial = np.append(terms_fiducial, 'bp'+str(i))
        if (curvature == 'F'):
            terms = np.delete(terms_fiducial, 1, axis=0)
            if (gamma == 'N'):
                terms = np.delete(terms_fiducial, [1, 8], axis=0)
        elif (curvature == 'NF'):
            terms = np.copy(terms_fiducial)
            if (gamma == 'N'):
                terms = np.delete(terms_fiducial, 8, axis=0)
        # Test on consistent number of parameters
        number_params = len(terms)
        if (curvature == 'F'):
            if (gamma == 'N'):
                # 7 cosmo parameters (no-gamma + F) = 7, 10 bias => should have 17 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error13 : different parameters not consistent in size')
                    sys.exit()
            elif (gamma == 'Y'):
                # 8 cosmo parameters (gamma + F) = 8, 10 bias => should have 18 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error14 : different parameters not consistent in size')
                    sys.exit()
        if (curvature == 'NF'):
            if (gamma == 'N'):
                # 8 cosmo parameters (no-gamma + NF) = 8, 10 bias => should have 18 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error15 : different parameters not consistent in size')
                    sys.exit()
            elif (gamma == 'Y'):
                # 9 cosmo parameters (gamma + NF) = 9, 10 bias => should have 19 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error16 : different parameters not consistent in size')
                    sys.exit()
    # Probe : WL
    if (probe == 'WL'):
        terms_fiducial = ['wm', 'wde', 'wb', 'w0', 'wa',
                          'h', 'ns', 's8', 'gamma', 'A_IA', 'n_IA', 'B_IA']
        # No bias with WL
        if ((num_bias_sp != 0) or (num_bias_ph != 0)):
            print('Error : WL has got no bias !')
            sys.exit()
        if (curvature == 'F'):
            terms = np.delete(terms_fiducial, 1, axis=0)
            if (gamma == 'N'):
                terms = np.delete(terms_fiducial, [1, 8], axis=0)
        elif (curvature == 'NF'):
            if (gamma == 'N'):
                terms = np.delete(terms_fiducial, 8, axis=0)
        # Test on consistent number of parameters
        number_params = len(terms)
        if (curvature == 'F'):
            if (gamma == 'N'):
                # 7 cosmo parameters (no-gamma + F) = 7, 3 IA => should have 10 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error17 : different parameters not consistent in size')
                    sys.exit()
            elif (gamma == 'Y'):
                # 8 cosmo parameters (gamma + F) = 8, 3 IA => should have 11 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error18 : different parameters not consistent in size')
                    sys.exit()
        if (curvature == 'NF'):
            if (gamma == 'N'):
                # 8 cosmo parameters (no-gamma + NF) = 8, 3 IA => should have 11 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error19 : different parameters not consistent in size')
                    sys.exit()
            elif (gamma == 'Y'):
                # 9 cosmo parameters (gamma + NF) = 9, 3 IA => should have 12 parameters
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error20 : different parameters not consistent in size')
                    sys.exit()
    # Probe : GCsp
    if (probe == 'GCsp'):

        # Defaut parameters : photo convention adopted from GCsp
        terms_fiducial = np.array(['wm', 'wde', 'wb', 'w0', 'wa', 'h', 'ns', 's8', 'gamma'])

        # Index ordering to apply on spectro ordering
        index_fiducial = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        
        # Add sig or None Remove None elements for terms_fiducial
        sig = np.array([sigp, sigv]) if case == 'PESS' else None
        terms_fiducial = np.append(terms_fiducial, sig)

        # Remove None elements
        terms_fiducial = terms_fiducial[terms_fiducial != np.array(None)]

        """
        # 2 Possibilities : 
        # 2.1) Adding specro (bias,P_shot) with num_bias_sp input
        # 2.2) print alpha parameter (alpha = b_spec/b_ph)  instead of bias
        if (num_args == 9) and (sys.argv[8] == '1'):
            for i in range(1, num_bias_sp+1):
              terms_fiducial = np.append(terms_fiducial, 'alpha'+str(i))
        elif (num_args == 9) and (sys.argv[8] == 'C'):
          for i in range(1, num_bias_sp+1):
            terms_fiducial = np.append(terms_fiducial, 'bs'+str(i))
            terms_fiducial = np.append(terms_fiducial, 'ps'+str(i))
        else: 
        """
        for i in range(1, num_bias_sp+1):
          terms_fiducial = np.append(terms_fiducial, 'bs'+str(i))
        for i in range(1, num_bias_sp+1):
          terms_fiducial = np.append(terms_fiducial, 'ps'+str(i))
        
        # GCsp Matrix possibly with FLAT/NON-FLAT and GAMMA/NO-GAMMA : find out a solution
        if (curvature == 'F'):
          terms = np.delete(terms_fiducial, 1, axis=0)
          if (gamma == 'N'):
            terms = np.delete(terms_fiducial, [1, 8], axis=0)
        elif (curvature == 'NF'):
          terms = np.copy(terms_fiducial)
          if (gamma == 'N'):
            terms = np.delete(terms_fiducial, 8, axis=0)

        # Test on consistent number of parameters
        number_params = len(terms)
        if (curvature == 'F'):
            if (gamma == 'N'):
                # 7 cosmo parameters (no-gamma + F) + (sig_p,sig_v) + num_bias_sp (for example 4xbias + 4xP_shot)
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error21 : different parameters not consistent in size')
                    sys.exit()
            elif (gamma == 'Y'):
                # 8 cosmo parameters (gamma + F) = 8
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error22 : different parameters not consistent in size')
                    sys.exit()
        if (curvature == 'NF'):
            if (gamma == 'N'):
                # 8 cosmo parameters (no-gamma + NF) = 8
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error23 : different parameters not consistent in size')
                    sys.exit()
            elif (gamma == 'Y'):
                # 9 cosmo parameters (gamma + NF) = 9
                
                print('size = ', size)
                print('number_params = ', number_params)
                diff_number_effective = size - number_params
                if (diff_number_effective != 0):
                    print('Error24 : different parameters not consistent in size')
                    sys.exit()

    # Check the inversion is correct
    C_new = np.linalg.pinv(C_in)
    #if (not np.allclose(np.diag(C_new.dot(C_in)), np.ones(len(C_in)))):
    #    print('Check : wrong inversion !')
    #    sys.exit()

    # Print each element of the matrix
    # in case FLAT and NON-FLAT

    # 2 Different cases : Flat and Non-Flat
    if (curvature == 'F'):
      print('')
      print(str(probe)+' FLAT CASE')
      print('')
      
      # Shift for FoM 
      delta_fom = 0

    elif (curvature == 'NF'):
      print('')
      print(str(probe)+' - NON-FLAT CASE')
      print('')

      # Shift for FoM 
      delta_fom = 1

    # Sub-block
    print(C_new)
    C_sub = [[C_new[2+delta_fom][2+delta_fom], C_new[2+delta_fom][3+delta_fom]],
            [C_new[3+delta_fom][2+delta_fom], C_new[3+delta_fom][3+delta_fom]]]
    print(C_sub)
    # Print constraint for each term
    i = 0
    while i < size:
        print(terms[i] + ' +/- ' + str(np.sqrt(C_new[i][i])))
        i = i+1

    # Compute FoM
    FOM = np.sqrt(1./(np.linalg.det(C_sub)))
    print('')
    print('FoM = ' + str(FOM))
    print('')

